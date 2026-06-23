import numpy as np
import pytest
from unittest.mock import MagicMock

from biobuddy.model_modifiers.modify_rom_based_on_moment_arm import (
    MuscleMomentArmAnalyzer,
)
from biobuddy import BiomechanicalModelReal
import biorbd
from pathlib import Path


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.nb_q = 2
    model.nb_muscles = 3
    model.dof_names = ["q1", "q2"]
    model.muscle_names = ["m1", "m2", "m3"]

    model.get_dof_ranges.return_value = np.array([[0, 3.14], [-3.14, 3.14]])

    return model


@pytest.fixture
def analyzer(fake_model):
    analyzer = MuscleMomentArmAnalyzer.__new__(MuscleMomentArmAnalyzer)

    analyzer.model = fake_model
    analyzer.muscle_moment_arm = np.array(
        [
            [
                [-1, -0.5, 0.5, 1, 1],
                [1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )

    analyzer.states = np.array([[0, 1, 2, 3, 3.14], [0, 0, 0, 0, 0]])

    analyzer.biorbd_model = MagicMock()
    analyzer.biorbd_model.musclesLengthJacobian.return_value.to_array.return_value = np.ones((3, 2))

    analyzer._ranges_by_joint = None

    return analyzer


def get_muscle_moment_analyzer():
    current_path_file = Path(__file__).parent.parent
    MODEL_EXAMPLE_PATH = f"{current_path_file}/examples/models/arm26_allbiceps_1dof.bioMod"
    MODEL_REAL = BiomechanicalModelReal().from_biomod(MODEL_EXAMPLE_PATH)
    return MuscleMomentArmAnalyzer(MODEL_EXAMPLE_PATH, MODEL_REAL)


def test_initialization():

    analyzer_example = get_muscle_moment_analyzer()

    assert isinstance(analyzer_example.model, BiomechanicalModelReal)
    assert isinstance(analyzer_example.biorbd_model, biorbd.Model)

    assert analyzer_example.model.nb_q > 0
    assert analyzer_example.model.nb_muscles > 0

    assert analyzer_example.accurate_ranges_array.shape == (
        analyzer_example.model.nb_q,
        2,
    )
    assert np.array_equal(
        analyzer_example.accurate_ranges_array,
        np.zeros((analyzer_example.model.nb_q, 2)),
    )

    assert analyzer_example.sign_lever_arm == {}


# ---------------------------
# Tests : sign_lever_arm
# ---------------------------


def test_create_sign_lever_arm_user_valid(analyzer):

    signs = np.array([[1, -1, 0], [0, 1, -1]])

    analyzer.sign_lever_arm = signs

    expected = {
        "q1": {"m1": 1, "m2": -1, "m3": 0},
        "q2": {"m1": 0, "m2": 1, "m3": -1},
    }

    # Compare values (Sign enum values)
    for q in expected:
        for m in expected[q]:
            assert analyzer.sign_lever_arm[q][m].value == expected[q][m]


def test_create_sign_lever_arm_user_wrong_shape(analyzer):

    with pytest.raises(ValueError):
        analyzer.sign_lever_arm = np.array([[1, 0]])


def test_create_sign_lever_arm_user_wrong_value(analyzer):

    with pytest.raises(ValueError):
        analyzer.sign_lever_arm = np.array([[2, 0, 1], [0, 1, -1]])


def test_sign_lever_arm_setter_dict_valid(analyzer):
    """Test that the sign_lever_arm setter accepts a valid dict."""
    from biobuddy.utils.enums import Sign

    analyzer._sign_lever_arm = {}
    signs_dict = {
        "q1": {"m1": 1, "m2": -1, "m3": 0},
        "q2": {"m1": 0, "m2": 1, "m3": -1},
    }
    analyzer.sign_lever_arm = signs_dict

    assert analyzer.sign_lever_arm["q1"]["m1"].value == 1
    assert analyzer.sign_lever_arm["q1"]["m2"].value == -1
    assert analyzer.sign_lever_arm["q2"]["m3"].value == -1


def test_sign_lever_arm_setter_dict_wrong_dof_keys(analyzer):
    """Test that dict with wrong dof keys raises ValueError."""
    analyzer._sign_lever_arm = {}
    with pytest.raises(ValueError, match="first keys"):
        analyzer.sign_lever_arm = {"wrong_dof": {"m1": 1, "m2": -1, "m3": 0}}


def test_sign_lever_arm_setter_dict_wrong_muscle_keys(analyzer):
    """Test that dict with wrong muscle keys raises ValueError."""
    analyzer._sign_lever_arm = {}
    with pytest.raises(ValueError, match="second keys"):
        analyzer.sign_lever_arm = {
            "q1": {"wrong_muscle": 1},
            "q2": {"m1": 0, "m2": 1, "m3": -1},
        }


def test_sign_lever_arm_setter_invalid_type(analyzer):
    """Test that non-dict, non-ndarray type raises ValueError."""
    with pytest.raises(ValueError):
        analyzer.sign_lever_arm = [[1, -1, 0], [0, 1, -1]]


# # ---------------------------
# # Tests : moment arm (mocked)
# # ---------------------------


def test_compute_moment_arm_ranges(analyzer):

    result = analyzer.compute_moment_arm_ranges()

    assert isinstance(result, dict)
    assert "q1" in result
    assert "q2" in result
    assert "m1" in result["q1"]

    # shape logique
    assert isinstance(result["q1"]["m1"], list)
    assert "range" in result["q1"]["m1"][0]
    assert "sign" in result["q1"]["m1"][0]


def test_ranges_by_joint_lazy_property(analyzer):

    # first call triggers computation
    r1 = analyzer.ranges_by_joint

    # second call uses cache
    r2 = analyzer.ranges_by_joint

    assert r1 is r2  # same object (cache)
    assert isinstance(r1, dict)


# # ---------------------------
# # Tests : zero finding
# # ---------------------------


def test_find_zero_newton(analyzer):
    q_init = np.array([0.5, 0.0])

    def fake_jac(q):
        class Obj:
            def to_array(self_inner):
                # zero at q[0] = 0
                return np.array([[q[0], 0], [q[0], 0], [q[0], 0]])

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jac

    q_star = analyzer.find_zero_newton(q_init, joint_id=0, muscle_id=0)

    assert q_star is not None
    assert abs(q_star[0]) < 1e-6


# # ---------------------------
# # Tests : compute_moment_arm_ranges
# # ---------------------------


# 1. Case: all zeros
# ---------------------------
def test_all_zero(analyzer):
    analyzer.muscle_moment_arm = np.zeros((2, 3, 5))

    result = analyzer.compute_moment_arm_ranges()

    dof_ranges = analyzer.model.get_dof_ranges()

    for i, q in enumerate(["q1", "q2"]):
        for m in ["m1", "m2", "m3"]:
            assert result[q][m] == [
                {
                    "range": (dof_ranges[0, i], dof_ranges[1, i]),
                    "sign": 0,
                }
            ]


# 2. Case: constant positive sign
# ---------------------------
def test_constant_positive(analyzer):
    analyzer.muscle_moment_arm = np.ones((2, 3, 5))

    def fake_jacobian(q):
        class Obj:
            def to_array(self_inner):
                return np.ones((3, 2))

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    for q in ["q1", "q2"]:
        for m in ["m1", "m2", "m3"]:
            ranges = result[q][m]
            assert len(ranges) == 1
            assert ranges[0]["sign"] == 1


# # 3. Case: sign change
# # ---------------------------
def test_sign_change(analyzer):

    analyzer.model.get_dof_ranges.return_value = np.array([[-2.0, -2.0], [2.0, 2.0]])

    analyzer.states = np.array([[0, 1, 2, 3, 3.14], [0, 0, 0, 0, 0]])

    analyzer.muscle_moment_arm = np.array(
        [
            [
                [-1, -0.5, 0.5, 1, 1],
                [1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )

    analyzer.find_zero_newton = lambda *args, **kwargs: np.array([1.5, 0.0])

    def fake_jacobian(q):
        class Obj:
            def to_array(self):
                return np.array(
                    [
                        [-1 if q[0] < 0 else 1, 0],
                        [-1 if q[0] < 0 else 1, 0],
                        [-1 if q[0] < 0 else 1, 0],
                    ]
                )

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    ranges = result["q1"]["m1"]

    assert len(ranges) >= 2
    assert {r["sign"] for r in ranges} == {-1, 1}


# # 4. Case: tolerance threshold
# # ---------------------------
def test_tolerance(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))

    small_values = 1e-8
    R = np.full((2, 3, 5), small_values)

    analyzer.muscle_moment_arm = R

    result = analyzer.compute_moment_arm_ranges(tol=1e-6)

    for q in ["q1", "q2"]:
        for m in ["m1", "m2", "m3"]:
            assert result[q][m][0]["sign"] == 0


# # 5. Output structure validation
# # ---------------------------
def test_output_structure(analyzer):
    analyzer.compute_joint_states = lambda n: np.zeros((2, 5))
    analyzer.compute_moment_arm = lambda states: np.ones((2, 3, 5))

    def fake_jacobian(q):
        class Obj:
            def to_array(self_inner):
                return np.ones((3, 2))

        return Obj()

    analyzer.biorbd_model.musclesLengthJacobian.side_effect = fake_jacobian

    result = analyzer.compute_moment_arm_ranges()

    assert isinstance(result, dict)

    for q in analyzer.model.dof_names:
        assert q in result
        for m in analyzer.model.muscle_names:
            assert m in result[q]
            assert isinstance(result[q][m], list)
            assert "range" in result[q][m][0]
            assert "sign" in result[q][m][0]


# # ---------
# # Tests : get
# # ---------
def test_get_ranges_from_dof_index(analyzer):
    sign_dict = {
        "q1": {"m1": [{"range": (-1, 0), "sign": -1}]},
        "q2": {"m2": [{"range": (0, 1), "sign": 1}]},
    }

    result = analyzer.get_ranges_from_dof_index(sign_dict, dof_idx=0)

    assert result == sign_dict["q1"]


def test_get_ranges_from_dof_and_muscle_indices(analyzer):
    sign_dict = {
        "q1": {
            "m1": [{"range": (-1, 0), "sign": -1}],
            "m2": [{"range": (0, 1), "sign": 1}],
        }
    }

    result = analyzer.get_ranges_from_dof_and_muscle_indices(sign_dict, dof_idx=0, idx_muscle=1)

    assert result == sign_dict["q1"]["m2"]


def test_get_ranges_from_dof_index_missing_dof(analyzer):
    """Test that a missing DOF raises a KeyError."""
    result_dict = {"q2": {"m1": []}}
    with pytest.raises(KeyError):
        analyzer.get_ranges_from_dof_index(result_dict, dof_idx=0)  # q1 is missing


# # ---------------------------
# # Tests : merge_ranges_joint
# # ---------------------------
def test_merge_ranges_joint(analyzer):

    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1.0, 0.0), "sign": -1}],
            "m2": [{"range": (0.0, 1.0), "sign": 1}],
            "m3": [{"range": (-0.5, 0.5), "sign": 1}],
        },
        "q2": {},
    }

    result = analyzer.merge_ranges_joint(dof_idx=0)

    expected = sorted([-1.0, 0.0, 1.0, -0.5, 0.5])

    assert result == expected


def test_merge_ranges_joint_duplicates(analyzer):
    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1.0, 0.0), "sign": -1}],
            "m2": [{"range": (0.0, 1.0), "sign": 1}],
            "m3": [{"range": (-1.0, 1.0), "sign": 1}],
        },
        "q2": {},
    }

    result = analyzer.merge_ranges_joint(dof_idx=0)

    # duplicates must be removed
    assert result == [-1.0, 0.0, 1.0]


# # ---------------------------
# # Tests : merge_ranges_joint empty case
# # ---------------------------
def test_merge_ranges_joint_empty(analyzer):
    analyzer._ranges_by_joint = {"q1": {}, "q2": {}}

    result = analyzer.merge_ranges_joint(dof_idx=0)

    assert result == []


# # -------------------
# # Tests : test accurate range
# # -------------------


# # Error: no sign_lever_arm
# # ---------------------------
def test_no_sign_lever_arm(analyzer):
    analyzer._sign_lever_arm = {}

    with pytest.raises(ValueError):
        analyzer.accurate_ranges_from_true_sign()


# # Basic filtering (correct case)
# # ---------------------------
def test_accurate_ranges_filtering(analyzer):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1, "m2": -1, "m3": 0},
        "q2": {"m1": 1, "m2": 1, "m3": -1},
    }

    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": 1}, {"range": (-1, 1), "sign": -1}],
            "m2": [{"range": (-1, 1), "sign": -1}],
            "m3": [{"range": (-1, 1), "sign": 0}],
        },
        "q2": {
            "m1": [{"range": (-1, 1), "sign": 1}],
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": -1}],
        },
    }

    analyzer.accurate_ranges_from_true_sign()

    result = analyzer.accurate_ranges_by_joint

    assert result["q1"]["m1"] == [{"range": (-1, 1), "sign": 1}]
    assert result["q1"]["m2"] == [{"range": (-1, 1), "sign": -1}]
    assert result["q1"]["m3"] == [{"range": (-1, 1), "sign": 0}]


# # Missing expected sign
# # ---------------------------
def test_missing_expected_sign(analyzer):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1, "m2": 1, "m3": 1},
        "q2": {"m1": 1, "m2": 1, "m3": 1},
    }

    analyzer._ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": -1}],  # mismatch
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": 1}],
        },
        "q2": {
            "m1": [{"range": (-1, 1), "sign": 1}],
            "m2": [{"range": (-1, 1), "sign": 1}],
            "m3": [{"range": (-1, 1), "sign": 1}],
        },
    }

    with pytest.warns(UserWarning):
        analyzer.accurate_ranges_from_true_sign()


# # -------------------
# # Tests : compare ranges
# # -------------------
def test_compare_ranges_no_difference(analyzer, capsys):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": 1}],
        }
    }

    diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    captured = capsys.readouterr()

    assert diff is False
    assert "Correct" in captured.out


# # compare_ranges: mismatch sign
# # ---------------------------
def test_compare_ranges_sign_mismatch(analyzer):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {
        "q1": {
            "m1": [{"range": (-1, 1), "sign": -1}],
        }
    }

    with pytest.warns(UserWarning, match="Sign mismatch"):
        diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    assert diff is True


# # compare_ranges: missing muscle
# # ---------------------------
def test_compare_ranges_missing_muscle(analyzer):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1},  # expected_sign != 0 -> should trigger warning
    }

    accurate_ranges = {"q1": {}}  # m1 is missing

    with pytest.warns(UserWarning, match="m1 missing in accurate_ranges"):
        diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)

    assert diff is True


# # compare_ranges: missing muscle with expected sign 0 (no warning)
# # ---------------------------
def test_compare_ranges_missing_muscle_sign_zero(analyzer):
    """Missing muscle is OK when expected sign is 0."""
    from biobuddy.utils.enums import Sign

    analyzer._sign_lever_arm = {
        "q1": {"m1": Sign.ZERO},
    }

    accurate_ranges = {"q1": {}}  # m1 is missing but sign is 0 -> no warning

    diff = analyzer.compare_ranges_and_user_sign(accurate_ranges)
    assert diff is False


# # compare_ranges: missing dof
# # ---------------------------
def test_compare_ranges_missing_dof(analyzer):
    analyzer._sign_lever_arm = {
        "q1": {"m1": 1},
    }

    accurate_ranges = {}

    with pytest.raises(ValueError, match="not found"):
        analyzer.compare_ranges_and_user_sign(accurate_ranges)


# # ---------------------------
# # Tests : create_accurate_rom
# # ---------------------------
def test_create_accurate_rom_basic(analyzer):
    analyzer.model.get_dof_ranges.return_value = np.array([[-1.0, -2.0], [1.0, 2.0]])

    analyzer.model.dof_names = ["q1", "q2"]
    analyzer.model.muscle_names = ["m1"]

    analyzer.accurate_ranges_array = np.zeros((2, 2))

    analyzer.accurate_ranges_by_joint = {
        "q1": {
            "m1": [{"range": (-0.5, 0.8), "sign": 1}],
        },
        "q2": {
            "m1": [{"range": (-1.5, 1.5), "sign": 1}],
        },
    }

    result = analyzer.create_accurate_rom()

    expected = np.array(
        [
            [-0.5, 0.8],
            [-1.5, 1.5],
        ]
    )

    assert result.shape == (2, 2)
    assert np.array_equal(result, expected)


def test_create_accurate_rom_empty_muscle_range_warns(analyzer):
    """create_accurate_rom should warn when a muscle has no accurate range."""
    analyzer.model.get_dof_ranges.return_value = np.array([[-1.0], [1.0]])
    analyzer.model.dof_names = ["q1"]
    analyzer.model.muscle_names = ["m1"]
    analyzer.accurate_ranges_array = np.zeros((1, 2))

    analyzer.accurate_ranges_by_joint = {
        "q1": {"m1": []},  # empty -> should trigger warning
    }

    with pytest.warns(UserWarning, match="No accurate range found"):
        result = analyzer.create_accurate_rom()

    assert result is not None


def test_create_accurate_rom_no_sign_lever_arm(analyzer):
    """create_accurate_rom returns None and prints message when no sign_lever_arm."""
    analyzer.accurate_ranges_by_joint = {}
    analyzer._sign_lever_arm = {}
    analyzer.accurate_ranges_array = np.zeros((2, 2))

    result = analyzer.create_accurate_rom()

    assert result is None


def test_create_accurate_rom_triggers_computation(analyzer):
    analyzer.accurate_ranges_by_joint = {}

    analyzer._sign_lever_arm = {
        "q1": {"m1": 1},
        "q2": {"m1": 1},
    }

    analyzer.model.dof_names = ["q1", "q2"]
    analyzer.model.muscle_names = ["m1"]

    analyzer.accurate_ranges_from_true_sign = lambda: None

    analyzer.model.get_dof_ranges.return_value = np.array([[-1.0, -2.0], [1.0, 2.0]])
    analyzer.accurate_ranges_array = np.zeros((2, 2))
    analyzer.accurate_ranges_by_joint = {
        "q1": {"m1": [{"range": (-0.5, 0.5), "sign": 1}]},
        "q2": {"m1": [{"range": (-0.5, 0.5), "sign": 1}]},
    }

    result = analyzer.create_accurate_rom()

    assert result is not None
    assert result.shape == (2, 2)


# # --------------------
# # Tests : get_correct_part_of_movement
# # --------------------


# get_correct_part_of_movement shape error
# ---------------------------
def test_get_correct_part_mvt_shape_error(analyzer):
    q = np.zeros((1, 10))  # wrong nb_q

    with pytest.raises(ValueError, match="Incorrect shape"):
        analyzer.get_correct_part_of_movement(q)


# # get_correct_part_of_movement all correct
# # ---------------------------
def test_get_correct_part_mvt_all_correct(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.0, 0.1, 0.2],
        ]
    )

    correct_idx, incorrect_idx, correct_q, incorrect_q = analyzer.get_correct_part_of_movement(q)

    assert len(correct_idx) == 1
    assert len(incorrect_idx) == 0


# # get_correct_part_of_movement all incorrect
# # ---------------------------
def test_get_correct_part_mvt_all_incorrect(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    )

    correct_idx, incorrect_idx, correct_q, incorrect_q = analyzer.get_correct_part_of_movement(q)

    assert len(correct_idx) == 0
    assert len(incorrect_idx) == 1


# # get_correct_part_of_movement auto-call ROM
# # ---------------------------
def test_get_correct_part_mvt_auto_rom(analyzer):
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.zeros((2, 2))

    analyzer.create_accurate_rom = lambda: np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    q = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.5],
        ]
    )

    result = analyzer.get_correct_part_of_movement(q)

    assert len(result) == 4


# # get_correct_part_of_movement mixed correct/incorrect segments
# # ---------------------------
def test_get_correct_part_mvt_mixed_segments(analyzer):
    """Verify that non-consecutive correct/incorrect frames are split into separate segments."""
    analyzer.model.nb_q = 2

    analyzer.accurate_ranges_array = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    # Frames: correct, incorrect, correct — results in 2 correct segments, 1 incorrect
    q = np.array(
        [
            [0.0, 2.0, 0.5],
            [0.0, 2.0, 0.5],
        ]
    )

    correct_idx, incorrect_idx, correct_q, incorrect_q = analyzer.get_correct_part_of_movement(q)

    assert len(correct_idx) == 2
    assert len(incorrect_idx) == 1


# # --------------------
# # Tests : plot_ranges_with_true_button
# # --------------------


def test_plot_ranges_with_true_button_structure():
    """Test the structure of the ranges plot without accurate ranges."""
    analyzer_example = get_muscle_moment_analyzer()
    fig = analyzer_example.plot_ranges_with_true_button(show_plot=False)

    # One subplot per DOF
    annotations = [a.text for a in fig.layout.annotations]
    assert len(annotations) == analyzer_example.model.nb_q
    dof_name = analyzer_example.model.dof_names[0]
    assert f"q0 - {dof_name}" in annotations

    # Title
    assert "sign_moment_arm_" in fig.layout.title.text

    # Without accurate_ranges, only "All ROM" button
    assert len(fig.layout.updatemenus[0].buttons) == 1
    assert fig.layout.updatemenus[0].buttons[0].label == "All ROM"

    # Two sign regions per muscle (positive + negative)
    assert len(fig.data) == analyzer_example.model.nb_muscles * 2

    # Check trace names and muscle assignment
    muscle_names_in_traces = [trace.y[0] for trace in fig.data]
    for muscle_name in analyzer_example.model.muscle_names:
        assert muscle_name in muscle_names_in_traces

    region_names = {trace.name for trace in fig.data}
    assert "Agonist region (+)" in region_names
    assert "Antagonist region (-)" in region_names


def test_plot_ranges_with_true_button_values():
    """Test trace values for the ranges plot."""
    analyzer_example = get_muscle_moment_analyzer()
    fig = analyzer_example.plot_ranges_with_true_button(show_plot=False)

    # First trace: BIClong antagonist region (negative sign, base at range min)
    assert fig.data[0].name == "Antagonist region (-)"
    assert fig.data[0].y == ("BIClong",)
    np.testing.assert_almost_equal(fig.data[0].base, 0.0, decimal=6)
    np.testing.assert_almost_equal(fig.data[0].x[0], 2.8697741103057655, decimal=6)

    # Second trace: BIClong agonist region (positive sign)
    assert fig.data[1].name == "Agonist region (+)"
    assert fig.data[1].y == ("BIClong",)
    np.testing.assert_almost_equal(fig.data[1].base, 2.8697741103057655, decimal=6)
    np.testing.assert_almost_equal(fig.data[1].x[0], 0.2718258896942345, decimal=6)

    # Third trace: BICshort antagonist region
    assert fig.data[2].name == "Antagonist region (-)"
    assert fig.data[2].y == ("BICshort",)
    np.testing.assert_almost_equal(fig.data[2].base, 0.0, decimal=6)
    np.testing.assert_almost_equal(fig.data[2].x[0], 2.8697741103057655, decimal=6)

    # Fourth trace: BICshort agonist region
    assert fig.data[3].name == "Agonist region (+)"
    assert fig.data[3].y == ("BICshort",)
    np.testing.assert_almost_equal(fig.data[3].base, 2.8697741103057655, decimal=6)
    np.testing.assert_almost_equal(fig.data[3].x[0], 0.2718258896942345, decimal=6)


def test_plot_ranges_with_true_button_with_accurate_ranges():
    """Test that the True ROM button is added when accurate_ranges_by_joint is set."""
    import warnings

    analyzer_example = get_muscle_moment_analyzer()

    # BIClong and BICshort are both antagonists (sign = -1) over most of the ROM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer_example.sign_lever_arm = np.array([[-1, -1]])
        analyzer_example.accurate_ranges_from_true_sign()

    fig = analyzer_example.plot_ranges_with_true_button(show_plot=False)

    # Should now have two buttons: "All ROM" and "True ROM"
    button_labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "All ROM" in button_labels
    assert "True ROM" in button_labels

    # Colors: traces in true ranges stay colored, others turn black
    true_rom_colors = fig.layout.updatemenus[0].buttons[1].args[0]["marker.color"]
    all_rom_colors = fig.layout.updatemenus[0].buttons[0].args[0]["marker.color"]
    assert len(true_rom_colors) == len(fig.data)
    assert len(all_rom_colors) == len(fig.data)
    assert "black" in true_rom_colors  # some traces are dimmed


# # --------------------
# # Tests : plot_q_qdot_rom
# # --------------------


def test_plot_q_qdot_rom_structure():
    """Test the structure of the q vs time plot."""
    analyzer_example = get_muscle_moment_analyzer()

    nb_dof = analyzer_example.model.nb_q  # 1
    t = np.linspace(0, 1, 10)
    q = np.array([np.linspace(0.5, 2.5, 10)])  # shape (1, 10)
    bounds = np.array([[0.0, 3.14]])

    all_correct_idx = [np.array([0, 1, 2]), np.array([6, 7, 8, 9])]
    all_incorrect_idx = [np.array([3, 4, 5])]

    fig = analyzer_example.plot_q_qdot_rom(t, q, bounds, all_correct_idx, all_incorrect_idx, show_plot=False)

    # Title
    assert "Joint states and ROM limits" in fig.layout.title.text

    # Subplot titles (dof names)
    annotations = [a.text for a in fig.layout.annotations]
    for dof_name in analyzer_example.model.dof_names:
        assert dof_name in annotations

    # Traces: 2 correct segments + 1 incorrect segment + 2 bounds = 5
    assert len(fig.data) == len(all_correct_idx) + len(all_incorrect_idx) + 2

    trace_names = [trace.name for trace in fig.data]
    assert trace_names.count("Usable moment range") == len(all_correct_idx)
    assert trace_names.count("Non usable moment range") == len(all_incorrect_idx)
    assert trace_names.count("Moment-consistent limits") == 2

    # Axis labels
    assert fig.layout.yaxis.title.text == "q (rad)"
    assert fig.layout.xaxis.title.text == "Time (s)"


def test_plot_q_qdot_rom_trace_data():
    """Test that trace x/y data matches the input arrays."""
    analyzer_example = get_muscle_moment_analyzer()

    t = np.linspace(0, 1, 6)
    q = np.array([np.linspace(1.0, 2.0, 6)])
    bounds = np.array([[0.5, 2.5]])

    all_correct_idx = [np.array([0, 1, 2])]
    all_incorrect_idx = [np.array([3, 4, 5])]

    fig = analyzer_example.plot_q_qdot_rom(t, q, bounds, all_correct_idx, all_incorrect_idx, show_plot=False)

    # First trace: correct segment
    correct_trace = fig.data[0]
    assert correct_trace.name == "Usable moment range"
    np.testing.assert_array_almost_equal(correct_trace.x, t[all_correct_idx[0]])

    # Second trace: incorrect segment
    incorrect_trace = fig.data[1]
    assert incorrect_trace.name == "Non usable moment range"
    np.testing.assert_array_almost_equal(incorrect_trace.x, t[all_incorrect_idx[0]])

    # Bound traces: constant y at the bound values
    lower_bound_trace = fig.data[2]
    upper_bound_trace = fig.data[3]
    assert lower_bound_trace.name == "Moment-consistent limits"
    np.testing.assert_array_almost_equal(lower_bound_trace.y, [bounds[0, 0]] * len(t))
    np.testing.assert_array_almost_equal(upper_bound_trace.y, [bounds[0, 1]] * len(t))

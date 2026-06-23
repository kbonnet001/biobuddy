import numpy as np
import biorbd

import warnings

from scipy.optimize import root_scalar
import copy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..validation.validate_muscles import MuscleValidator
from ..utils.enums import Sign
from ..components.real.biomechanical_model_real import BiomechanicalModelReal


class MuscleMomentArmAnalyzer(MuscleValidator):
    def __init__(
        self,
        model_path: str,
        model: BiomechanicalModelReal,
        nb_states: int = 50,
        custom_ranges: np.ndarray | None = None,
    ):
        """
        Initialize the MuscleMomentArmAnalyzer.

        Parameters
        ----------
        model_path : str
            Path to the .bioMod file describing the musculoskeletal model.
        model : BiomechanicalModelReal
            The loaded biomechanical model.
        nb_states : int
            The number of state increments to consider while making the plots.
        custom_ranges : np.ndarray | None
            Optional custom ranges for the degrees of freedom. If None, the default ranges from the model are used.
        """

        super().__init__(model=model, nb_states=nb_states, custom_ranges=custom_ranges)
        self.biorbd_model = biorbd.Model(model_path)  # TODO: make genrelazable to other engines
        self._sign_lever_arm = {}
        self._ranges_by_joint = None  # Lazy evaluation
        self.accurate_ranges_by_joint = {}
        self.accurate_ranges_array = np.zeros((self.model.nb_q, 2))

    @property
    def sign_lever_arm(self) -> dict[str, dict[str, Sign]]:
        return self._sign_lever_arm

    @sign_lever_arm.setter
    def sign_lever_arm(self, list_sign: dict[str, dict[str, int | Sign]] | np.ndarray[int | Sign]):
        if isinstance(list_sign, dict):
            if set(list_sign.keys()) != set(self.model.dof_names):
                raise ValueError(
                    f"The first keys of the dict should be the dof names, got {list_sign.keys()}, expected {self.model.dof_names}"
                )
            for dof_name in self.model.dof_names:
                if set(list_sign[dof_name].keys()) != set(self.model.muscle_names):
                    raise ValueError(
                        f"The second keys of the dict should be the muscle names, got {list_sign[dof_name].keys()} for dof {dof_name}, expected {self.model.muscle_names}"
                    )
                else:
                    self._sign_lever_arm[dof_name] = {}
                    for m_name in self.model.muscle_names:
                        sign = list_sign[dof_name][m_name]
                        if isinstance(sign, Sign):
                            self._sign_lever_arm[dof_name][m_name] = sign
                        else:
                            if sign not in [s.value for s in Sign]:
                                raise ValueError(
                                    f"Invalid sign {sign} at ({dof_name}, {m_name}). Sign must be either -1, 0 or 1"
                                )
                            else:
                                self._sign_lever_arm[dof_name][m_name] = Sign(sign)
        elif isinstance(list_sign, np.ndarray):
            if list_sign.shape != (self.model.nb_q, self.model.nb_muscles):
                raise ValueError(
                    f"Invalid shape, need : ({self.model.nb_q},{self.model.nb_muscles}) but have {list_sign.shape}"
                )
            self._sign_lever_arm = {
                dof_name: {m_name: None for m_name in self.model.muscle_names} for dof_name in self.model.dof_names
            }
            for idx_dof, dof_name in enumerate(self.model.dof_names):
                for idx_muscle, m_name in enumerate(self.model.muscle_names):
                    sign = list_sign[idx_dof, idx_muscle]
                    if sign not in [s.value for s in Sign]:
                        raise ValueError(
                            f"Invalid sign {sign} at ({dof_name}, {m_name}). Sign must be either -1, 0 or 1"
                        )
                    else:
                        self._sign_lever_arm[dof_name][m_name] = Sign(sign)
        else:
            raise ValueError(
                f"list_sign should be a dict[dict[int | Sign]] or a np.array[int | Sign], you have {type(list_sign)}."
            )

    def find_zero_newton(self, q_init: np.ndarray, joint_id: int, muscle_id: int):
        """
        Found q* such that r_{joint_id, muscle_id}(q*) = 0
        by only modifying joint_id (Newton 1D).

        Parameters
        ----------
        q_init : (n_q,)
        joint_id : joint index
        muscle_id : muscle index
        """
        q = q_init.copy()

        def f(x):
            q_local = q.copy()
            q_local[joint_id] = x
            J = self.biorbd_model.musclesLengthJacobian(q_local).to_array()
            return J[muscle_id, joint_id]

        sol = root_scalar(f, method="newton", x0=q[joint_id])

        if sol.converged:
            q[joint_id] = sol.root
            return q
        return None

    @property
    def ranges_by_joint(self):
        """
        Lazily compute and cache the ranges_by_joint property.
        """
        if self._ranges_by_joint is None:
            self._ranges_by_joint = self.compute_moment_arm_ranges()
        return self._ranges_by_joint

    def compute_moment_arm_ranges(self, tol: float = 1e-6) -> dict[str, dict[str, tuple[int]]]:
        """
        Directly returns the intervals of constant sign for the moment arm.

        Returns
        -------
        dict[j][m] -> list of {"range": (a,b), "sign": ±1}
        """
        if not hasattr(self, "muscle_moment_arm") or not isinstance(self.muscle_moment_arm, np.ndarray):
            raise AttributeError(
                "self.muscle_moment_arm must be initialized with MuscleValidator before using this method."
            )
        if not hasattr(self, "states") or not isinstance(self.states, np.ndarray) or self.states.ndim < 2:
            raise AttributeError(
                "self.states must be initialized with MuscleValidator, be a NumPy array, and have at least 2 dimensions."
            )

        self.muscle_moment_arm[np.abs(self.muscle_moment_arm) < tol] = 0.0

        result = {
            dof_name: {muscle_name: [] for muscle_name in self.model.muscle_names} for dof_name in self.model.dof_names
        }

        n_q = self.model.nb_q
        n_m = self.model.nb_muscles

        for dof_idx in range(n_q):
            for idx_muscle in range(n_m):
                ranges = []

                if np.all(np.abs(self.muscle_moment_arm[dof_idx, idx_muscle]) < tol):
                    a = self.model.get_dof_ranges()[0, dof_idx]
                    b = self.model.get_dof_ranges()[1, dof_idx]
                    ranges.append({"range": (a, b), "sign": 0})

                else:

                    # 1. Detect indices where the sign of the moment arm changes between consecutive states
                    prod = (
                        self.muscle_moment_arm[dof_idx, idx_muscle, :-1]
                        * self.muscle_moment_arm[dof_idx, idx_muscle, 1:]
                    )
                    flip_idx = np.where(prod < 0)[0]

                    # 2. Newton --> find zeros more precisely
                    zeros = []
                    for i in flip_idx:
                        q_star = self.find_zero_newton(self.states[:, i], dof_idx, idx_muscle)
                        if q_star is not None:
                            zeros.append(q_star[dof_idx])

                    zeros = sorted(zeros)

                    # 3. bounds
                    bounds = (
                        [self.model.get_dof_ranges()[0, dof_idx]] + zeros + [self.model.get_dof_ranges()[1, dof_idx]]
                    )

                    # 4. Determine the sign of the moment arm in each interval between detected zero crossings
                    for a, b in zip(bounds[:-1], bounds[1:]):

                        mid = 0.5 * (a + b)

                        q_test = np.zeros(n_q)
                        q_test[dof_idx] = mid

                        r = self.biorbd_model.musclesLengthJacobian(q_test).to_array()[idx_muscle, dof_idx]
                        s = int(np.sign(r))

                        # Explicitly handle zero sign intervals for robustness
                        ranges.append({"range": (a, b), "sign": s})

                result[self.model.dof_names[dof_idx]][self.model.muscle_names[idx_muscle]] = ranges

        return result

    def merge_ranges_joint(self, dof_idx: int):
        """
        ranges_joint : dict[muscle_idx] -> list of {"range": (a,b), "sign": ±1}

        Returns
        -------
        merged_bounds : list of float
            Unique sorted bounds of all intervals from all muscles
        """
        unique_bounds = set()

        ranges = self.get_ranges_from_dof_index(self.ranges_by_joint, dof_idx)
        for muscle_ranges in ranges.values():
            for r in muscle_ranges:
                a, b = r["range"]
                unique_bounds.add(a)
                unique_bounds.add(b)

        merged_bounds = sorted(unique_bounds)
        return merged_bounds

    def get_ranges_from_dof_index(self, sign_dict: dict, dof_idx: int):
        """
        Retrieve ranges for a specific DOF index from a sign dictionary.
        Parameters
        ----------
        sign_dict : dict[j][m] -> list of {"range": (a,b), "sign": ±1}
        dof_idx : index of the DOF

        """
        dof_name = self.model.dof_names[dof_idx]
        if dof_name not in sign_dict:
            raise KeyError(f"DOF name '{dof_name}' not found in sign_dict.")
        return sign_dict[dof_name]

    def get_ranges_from_dof_and_muscle_indices(self, sign_dict: dict, dof_idx: int, idx_muscle: int):
        """
        Retrieve ranges for a specific DOF and muscle index from a sign dictionary.
        Parameters
        ----------
        sign_dict : dict[j][m] -> list of {"range": (a,b), "sign": ±1}
        dof_idx : index of the DOF
        idx_muscle : index of the muscle
        """
        dof_name = self.model.dof_names[dof_idx]
        muscle_name = self.model.muscle_names[idx_muscle]
        return sign_dict[dof_name][muscle_name]

    def accurate_ranges_from_true_sign(self):
        if self.sign_lever_arm == {}:
            raise ValueError(
                "No sign_lever_arm. Please, use self.sign_lever_arm = np.ndarray[int | Sign] | dict[str, dict[str, int | Sign]] to create it."
            )
        accurate_ranges = copy.deepcopy(self.ranges_by_joint)

        for dof_name in self.model.dof_names:
            for m_name in self.model.muscle_names:

                all_items = accurate_ranges[dof_name][m_name]
                expected_sign = self.sign_lever_arm[dof_name][m_name]

                if expected_sign not in [item["sign"] for item in all_items]:
                    warnings.warn(f"There is no range with the sign {expected_sign} " f"for {dof_name} {m_name}")
                else:
                    accurate_ranges[dof_name][m_name] = [item for item in all_items if item["sign"] == expected_sign]

        print("\nComparison with user sign : ")
        self.compare_ranges_and_user_sign(accurate_ranges)

        self.accurate_ranges_by_joint = accurate_ranges

    def compare_ranges_and_user_sign(self, accurate_ranges: dict[str, dict[str, dict[str, tuple[int] | int]]]):
        """
        Compare the accurate ranges with the user-provided signs and raise errors if there are mismatches.
        Parameters
        ----------
        accurate_ranges : dict[j][m] -> list of {"range": (a,b), "sign": ±1}
             Ranges computed by accurate_ranges_from_true_sign method
        Returns
        -------
        bool
            True if there is a difference between accurate_ranges and user signs, False otherwise.
        """
        difference = False
        for dof_name, muscles_sign in self.sign_lever_arm.items():

            # Check if dof_name is in accurate_ranges
            if dof_name not in accurate_ranges:
                raise ValueError(f"{dof_name} not found in accurate_ranges")

            for m_name, expected_sign in muscles_sign.items():

                # Muscle is missing in accurate_ranges[dof_name]
                if m_name not in accurate_ranges[dof_name]:

                    expected_sign_value = expected_sign.value if isinstance(expected_sign, Sign) else expected_sign
                    if expected_sign_value == 0:
                        # ok
                        continue
                    else:
                        warnings.warn(
                            f"{m_name} missing in accurate_ranges[{dof_name}] " f"but expected sign is {expected_sign}"
                        )
                        difference = True
                        continue

                # Muscle is present in accurate_ranges[dof_name], check if expected_sign is among the available signs
                ranges = accurate_ranges[dof_name][m_name]
                available_signs = {item["sign"] for item in ranges}

                if expected_sign not in available_signs:
                    warnings.warn(
                        f"Sign mismatch for {dof_name}-{m_name}: "
                        f"expected user sign {expected_sign}, "
                        f"available {available_signs}"
                    )
                    difference = True
        if not difference:
            print("Correct")
        return difference

    def create_accurate_rom(self) -> np.ndarray:
        """
        Create an accurate ROM array for each DOF based on the accurate_ranges_by_joint.
        This method computes the intersection of the ranges for all muscles for each DOF and stores the result
        in accurate_ranges_array.

        Returns
        -------
        np.ndarray
            The accurate ROM array for each DOF (accurate_ranges_array).
        """
        if self.accurate_ranges_by_joint == {}:
            if self.sign_lever_arm == {}:
                print(
                    "No sign_lever_arm. Please, use either create_sign_lever_arm_user() or update_sign_lever_arm to create it"
                )
                return None
            else:
                self.accurate_ranges_from_true_sign()

        for dof_idx, dof_name in enumerate(self.model.dof_names):
            rom_range = np.array(
                [
                    self.model.get_dof_ranges()[0, dof_idx],
                    self.model.get_dof_ranges()[1, dof_idx],
                ]
            )
            for m_name in self.model.muscle_names:
                if self.accurate_ranges_by_joint[dof_name][m_name]:
                    rom_range[0] = max(
                        rom_range[0],
                        self.accurate_ranges_by_joint[dof_name][m_name][0]["range"][0],
                    )
                    rom_range[1] = min(
                        rom_range[1],
                        self.accurate_ranges_by_joint[dof_name][m_name][0]["range"][1],
                    )
                else:
                    warnings.warn(
                        f"No accurate range found for {dof_name}, {m_name}. Skipping this muscle in ROM calculation."
                    )

            self.accurate_ranges_array[dof_idx, :] = rom_range
        return self.accurate_ranges_array

    def get_correct_part_of_movement(self, q: np.ndarray):
        """
        Given a movement q(t), return the indices and values of the segments of q(t) that are consistent with the
        accurate ROM.
        Parameters
        ----------
        q : np.ndarray
            A 2D array of shape (nb_q, N) representing the movement over time, where nb_q is the number of DOFs and
            N is the number of time steps.
        Returns
        -------
        tuple
            A tuple containing:
            - all_correct_idx: List of numpy arrays of indices for the segments of q(t) that are consistent with
              the accurate ROM.
            - all_incorrect_idx: List of numpy arrays of indices for the segments of q(t) that are not consistent with
              the accurate ROM.
            - all_correct_q: List of numpy arrays of q values (shape: (nb_q, segment_length)) for the segments that are consistent with
              the accurate ROM.
            - all_incorrect_q: List of numpy arrays of q values (shape: (nb_q, segment_length)) for the segments that are not consistent with
              the accurate ROM.
        """

        if q.shape[0] != self.model.nb_q:
            raise ValueError(f"Incorrect shape, must have {self.model.nb_q} but got {q.shape[0]}")

        if np.allclose(self.accurate_ranges_array, np.zeros((self.model.nb_q, 2))):
            self.create_accurate_rom()

        N = q.shape[1]

        idx_correct = []
        idx_incorrect = []
        # sort idx
        for n in range(N):
            is_correct = np.all(
                (self.accurate_ranges_array[:, 0] <= q[:, n]) & (q[:, n] <= self.accurate_ranges_array[:, 1])
            )
            if is_correct:
                idx_correct.append(n)
            else:
                idx_incorrect.append(n)

        idx_correct = np.array(idx_correct)
        idx_incorrect = np.array(idx_incorrect)

        # split
        def split_consecutive(idx: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Split an array of indices into consecutive segments.
            Parameters
            ----------
            idx : np.ndarray
                An array of indices to be split into consecutive segments.
            Returns
            -------
            tuple
                A tuple containing:
                - List of numpy arrays of indices for each consecutive segment.
                - List of numpy arrays of q values (shape: (nb_q, segment_length))
                for each consecutive segment.
            """
            if len(idx) == 0:
                return [], []
            splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            all_q = []
            for split in splits:
                mvt = np.zeros((self.model.nb_q, len(split)))
                for k in range(len(split)):
                    mvt[:, k] = q[:, k]
                all_q.append(mvt)
            return splits, all_q

        all_correct_idx, all_correct_q = split_consecutive(idx_correct)
        all_incorrect_idx, all_incorrect_q = split_consecutive(idx_incorrect)

        return all_correct_idx, all_incorrect_idx, all_correct_q, all_incorrect_q

    def plot_ranges_with_true_button(self, path_to_save: str = "", show_plot: bool = True) -> None:

        n_q = len(self.ranges_by_joint)
        nb_line, nb_column = n_q, 1
        fig = make_subplots(
            rows=nb_line,
            cols=nb_column,
            subplot_titles=[f"q{idx} - {dof_name}" for idx, dof_name in enumerate(self.ranges_by_joint)],
        )

        legend_added = {"pos": False, "neg": False, "zero": False}
        trace_indices_true = []

        for dof_idx, dof_name in enumerate(self.ranges_by_joint):
            row, col = dof_idx + 1, 1
            muscles = list(self.ranges_by_joint[dof_name].keys())
            component_names = list(self.ranges_by_joint[dof_name].keys())

            for m_name in muscles:
                all_ranges = self.ranges_by_joint[dof_name][m_name]
                true_ranges = self.accurate_ranges_by_joint.get(dof_name, {}).get(m_name, [])

                for r in all_ranges:
                    a, b = r["range"]
                    sign = r["sign"]
                    width = b - a

                    if self.accurate_ranges_by_joint != {}:

                        is_true = any(tr["range"] == r["range"] and tr["sign"] == r["sign"] for tr in true_ranges)
                        trace_indices_true.append(is_true)
                    else:
                        is_true = True

                    if sign > 0:
                        color = "rgba(214,39,40,0.6)"
                        pattern = "+"
                        legend_key = "pos"
                        name = "Agonist region (+)"
                    elif sign < 0:
                        color = "rgba(44,160,44,0.6)"
                        pattern = "-"
                        legend_key = "neg"
                        name = "Antagonist region (-)"
                    else:
                        color = "rgba(200,200,200,0.6)"
                        pattern = ""
                        legend_key = "zero"
                        name = "Zero"

                    show_legend = is_true and not legend_added[legend_key]
                    if show_legend:
                        legend_added[legend_key] = True

                    trace = go.Bar(
                        x=[width],
                        y=[m_name],
                        base=a,
                        orientation="h",
                        marker=dict(
                            color=color,
                            line=dict(color="black", width=1),
                            pattern=dict(shape=pattern),
                        ),
                        name=name,
                        showlegend=show_legend,
                    )
                    fig.add_trace(trace, row=row, col=col)

            fig.update_xaxes(title_text="Range (rad)", row=row, col=col)

            fig.update_yaxes(
                categoryorder="array",
                categoryarray=component_names[::-1],
                tickmode="array",
                tickvals=component_names,
                ticktext=component_names,
                row=row,
                col=col,
            )

        buttons = [
            dict(
                label="All ROM",
                method="update",
                args=[
                    {"marker.color": [fig.data[i].marker.color for i in range(len(fig.data))]},
                    {"title": "All ROM"},
                ],
            ),
        ]

        if self.accurate_ranges_by_joint != {}:
            buttons.append(
                dict(
                    label="True ROM",
                    method="update",
                    args=[
                        {
                            "marker.color": [
                                (fig.data[i].marker.color if trace_indices_true[i] else "black")
                                for i in range(len(fig.data))
                            ]
                        },
                        {"title": "True ROM"},
                    ],
                ),
            )
        title = f"sign_moment_arm_{path_to_save.replace('/', '_')}"
        fig.update_layout(
            title=title,
            barmode="overlay",
            updatemenus=[dict(type="buttons", showactive=True, buttons=buttons)],
            height=300 * nb_line,
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                orientation="v",
                traceorder="normal",
            ),
        )
        if path_to_save:
            fig.write_html(f"{path_to_save}/Sign_moment_arm.html")
        if show_plot:
            fig.show(renderer="browser")

    def plot_q_qdot_rom(
        self,
        t: np.ndarray,
        q: np.ndarray,
        bounds: np.ndarray,
        all_correct_idx: np.ndarray,
        all_incorrect_idx: np.ndarray,
        path_to_save: str = "",
        show_plot: bool = True,
    ) -> None:

        nb_dof = q.shape[0]

        fig = make_subplots(
            rows=nb_dof,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=self.model.dof_names,
        )

        for idx_dof in range(nb_dof):
            row = idx_dof + 1

            for k, seg in enumerate(all_correct_idx):
                fig.add_trace(
                    go.Scatter(
                        x=t[seg],
                        y=q[idx_dof, seg],
                        mode="lines",
                        line=dict(color="blue"),
                        name="Usable moment range",
                        showlegend=(idx_dof == 0 and k == 0),
                    ),
                    row=row,
                    col=1,
                )

            for k, seg in enumerate(all_incorrect_idx):
                fig.add_trace(
                    go.Scatter(
                        x=t[seg],
                        y=q[idx_dof, seg],
                        mode="lines",
                        line=dict(color="red", dash="dashdot"),
                        name="Non usable moment range",
                        showlegend=(idx_dof == 0 and k == 0),
                    ),
                    row=row,
                    col=1,
                )

            # -------- bounds --------
            for j in range(2):
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=[bounds[idx_dof, j]] * len(t),
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name="Moment-consistent limits",
                        showlegend=(idx_dof == 0 and j == 0),
                    ),
                    row=row,
                    col=1,
                )

            fig.update_yaxes(title_text="q (rad)", row=row, col=1)

        transition_indices = []

        for seg_list in [all_correct_idx, all_incorrect_idx]:
            for seg in seg_list:
                transition_indices.append(seg[0])
                transition_indices.append(seg[-1])

        for idx in np.unique(transition_indices):
            fig.add_vline(
                x=t[idx],
                line=dict(color="gray", dash="dash"),
            )

        for i in range(1, nb_dof):
            fig.update_xaxes(showticklabels=False, row=i, col=1)

        fig.update_xaxes(title_text="Time (s)", row=nb_dof, col=1)

        fig.update_layout(
            height=300 * nb_dof,
            title=f"Joint states and ROM limits\n{path_to_save.replace('/', '_')}",
            template="plotly_white",
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
        )
        if path_to_save:
            fig.write_html(f"{path_to_save}/Joint_states_and_ROM_limits.html")
        if show_plot:
            fig.show(renderer="browser")

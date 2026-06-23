"""
This script is an example to how to keep only the correct range of motion where the moment arm is correct.
Using the classe MuscleMomentArmAnalyzer, this script permite to found all section where the moment arm have
different sign.
Then, the user could give the correct sign of moment arm for each joint and muscle.
The correct rom are then kept.
Then, user could give a mouvement q(t), where q(t_i) could be outside the correct rom.
This mouvement cut to keep only sections where the moment arm is correct.

TODO: This example should also show how to modify the range of motion of a model based on the muscle moment arm analysis.
"""

from biobuddy import BiomechanicalModelReal, Sign
from biobuddy.model_modifiers.modify_rom_based_on_moment_arm import (
    MuscleMomentArmAnalyzer,
)
import numpy as np
from pathlib import Path

# 1) Create a moment arm analyzer for a model
# --------------------------------------------

# Path to the model to check
current_path_file = Path(__file__).parent
model_path = f"{current_path_file}/models/arm26_allbiceps_1dof.bioMod"
path_to_save = f"{current_path_file}/data"

# Load the model
model = BiomechanicalModelReal().from_biomod(model_path)

# Create the moment arm analyzer
moment_arm_analyser = MuscleMomentArmAnalyzer(model_path, model, nb_states=50)

# You can now access all moment arm sign ranges for each DOF/muscle
# Each range corresponds to either a negative or positive sign
# If the sign is 0, the moment arm is null over the entire ROM
print(moment_arm_analyser.ranges_by_joint)
# ----
# Get the ranges for a specific DoF and muscle
print(moment_arm_analyser.ranges_by_joint["r_ulna_radius_hand_rotation1_rotZ"]["BIClong"])

# Alternative way using indices
print(
    moment_arm_analyser.get_ranges_from_dof_and_muscle_indices(
        moment_arm_analyser.ranges_by_joint, dof_idx=0, idx_muscle=0
    )
)
# ----
# Get all muscle ranges for one DOF
print(moment_arm_analyser.ranges_by_joint["r_ulna_radius_hand_rotation1_rotZ"])

# Alternative way using DoF index only
print(moment_arm_analyser.get_ranges_from_dof_index(moment_arm_analyser.ranges_by_joint, dof_idx=0))
# ----
# Visualize the computed ranges
moment_arm_analyser.plot_ranges_with_true_button(
    path_to_save=path_to_save,
)

# 2) Specify the expected sign of the moment arms
# ------------------------------------------------

# You can define the expected moment arm sign in two ways:

# a) Using a NumPy array:
# One row per DOF, one column per muscle
sign_lever_arm_user = np.array(
    [
        [-1, -1],
    ]
)
moment_arm_analyser.sign_lever_arm = sign_lever_arm_user

# b) Using an explicit dictionary:
# (Note: this reflects the internal data structure)
sign_lever_arm_user = {
    "r_ulna_radius_hand_rotation1_rotZ": {
        "BIClong": Sign.NEGATIVE,
        "BICshort": Sign.NEGATIVE,
    },
}
moment_arm_analyser.sign_lever_arm = sign_lever_arm_user

# Filter the ranges based on the expected sign
# and keep only the consistent ranges
moment_arm_analyser.accurate_ranges_from_true_sign()
print(moment_arm_analyser.accurate_ranges_by_joint)

# Generate the final RoM array from the filtered ranges
accurate_ranges = moment_arm_analyser.create_accurate_rom()
print(accurate_ranges)

# Visualize the filtered ranges
moment_arm_analyser.plot_ranges_with_true_button(
    path_to_save=path_to_save,
)

# 3) Extract usable q(t)
# -----------------------

# Provide q(t) and retrieve only the valid (usable) segments

# Create an arbitrary movement
N = 100
q = np.zeros((model.nb_q, N))
for idx_q in range(model.nb_q):
    q[idx_q, :] = np.linspace(0.0, np.pi, N)

# Get indices and values of valid and invalid portions of q(t)
all_correct_idx, all_incorrect_idx, all_correct_q, all_incorrect_q = moment_arm_analyser.get_correct_part_of_movement(q)

# Visualize q(t) together with RoM and valid/invalid segments
moment_arm_analyser.plot_q_qdot_rom(
    np.linspace(0.0, 1, N),
    q,
    moment_arm_analyser.accurate_ranges_array,
    all_correct_idx,
    all_incorrect_idx,
    path_to_save=path_to_save,
)

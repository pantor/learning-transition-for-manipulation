from pathlib import Path

import numpy as np

from cfrankr import Affine
from picking.epoch import Epoch
from picking.param import Bin, Mode, SelectionMethod


class Config:
    start_bin = Bin.Right
    mode = Mode.Measure

    # Camera
    camera_suffixes = ['ed']

    grasp_database = 'cylinder-cube-1'
    shift_database = 'shifting'

    # grasp_model = ('cylinder-cube-1', 'model-6-arch-more-layer')
    grasp_model = ('small-cubes-2', 'model-1-types')
    model_input_suffixes = ['ed']

    # grasp_model = ('cylinder-screw-3', 'model-7')
    # model_input_suffixes = ['ed', 'rd', 'rc']


    # grasp_model = ('cylinder-cube-2', 'model-1-lateral')
    # grasp_model = ('cylinder-cube-mc-1', 'model-2')
    # grasp_model = ('cylinder-cube-1', 'model-7-side')
    # grasp_model = ('cylinder-cube-1', 'model-6-arch-more-layer')
    # grasp_model = ('adversarial', 'model-4-g')
    # grasp_model = ('small-cubes-2', 'model-2-types')
    # grasp_model = ('screw-2', 'model-1-screw')
    # grasp_model = ('cube-1', 'ann-model-2')  # For specific grasps
    # grasp_model = ('cylinder-1', 'model-specific-3')

    shift_model = ('shifting', 'model-1')

    # Epochs
    epochs = [
        Epoch(
            number_episodes=1000,
            selection_method=SelectionMethod.Top5LowerBound,
            percentage_secondary=0.0,
            secondary_selection_method=SelectionMethod.Random,
        )
    ]

    # General structure
    change_bins = False
    bin_empty_at_max_probability = 0.3
    shift_objects = False
    file_objects = False
    set_zero_reward = False
    home_gripper = True

    # Images
    take_after_images = False
    take_direct_images = False
    take_lateral_images = False
    predict_images = False

    # Overview image
    lateral_overview_image = False
    overview_image_angles = [(-0.6, 0.0), (-0.3, 0.0), (0.3, 0.0), (0.6, 0.0)]

    # Lateral images
    # lateral_images_angles = [(b, 0) for b in np.linspace(-0.6, 0.6, 7)]
    lateral_images_angles = [(b, c) for b in np.linspace(-0.6, 0.6, 5) for c in np.linspace(-0.6, 0.6, 5)]

    # URL
    database_url = 'http://127.0.0.1:8080/api/'

    # Bin
    box = {
        'center': [-0.001, -0.0065, 0.372],  # [m]
        'size': [0.172, 0.281, 0.068],  # [m]
    }

    # Distances
    image_distance_from_pose = 0.350  # [m]
    default_image_pose = Affine(z=-image_distance_from_pose)
    approach_distance_from_pose = 0.120 if mode != Mode.Perform else 0.075  # [m]
    lower_random_pose = [-0.07, -0.12, 0.0, -1.4, 0.0, 0.0]  # [m, rad]
    upper_random_pose = [0.07, 0.12, 0.0, 1.4, 0.0, 0.0]  # [m, rad]

    # Dynamics and forces
    general_dynamics_rel = 0.32 if mode != Mode.Perform else 0.9
    gripper_speed = 0.06 if mode != Mode.Perform else 0.1  # [m/s]

    # Model training
    train_model = False
    train_async = True
    train_model_every_number_cycles = 10
    train_script = Path.home() / 'Documents' / 'bin_picking' / 'scripts' / 'learning' / 'monte_carlo.py'

    # Grasping
    grasp_type = 'DEFAULT'  # DEFAULT, SPECIFIC, TYPE
    check_grasp_second_time = False
    adjust_grasp_second_time = False
    change_bin_at_number_of_failed_grasps = 12  # 10-15 Normal
    release_during_grasp_action = True
    release_in_other_bin = True
    release_as_fast_as_possible = True
    random_pose_before_release = True
    max_random_affine_before_release = Affine(0.055, 0.10, 0.0, 1.2)  # [m, rad]
    move_down_distance_for_release = 0.11  # [m]
    measurement_gripper_force = 20.0  # [N], 15
    performance_gripper_force = 40.0  # [N]

    gripper_classes = [0.05, 0.07, 0.086]  # [m]
    grasp_z_offset = 0.015  # [m]

    place_z_offset = -0.014  # [m]


    if 'screw' in grasp_model[0]:
        gripper_classes = [0.025, 0.05, 0.07, 0.086]  # [m]
        grasp_z_offset = 0.008  # [m]


    # Evaluation
    evaluation_path = Path.home() / 'Documents' / 'data' / 'cylinder-cube-1' / 'evaluation' / 'grasp-rate-prediction-step.txt'
    change_bin_at_number_of_success_grasps = 11
    number_objects_in_bin = 20

    # Shift
    grasp_shift_threshold = 0.6
    shift_empty_threshold = 0.25  # default: 0.29
    shift_distance = 0.03  # [m]

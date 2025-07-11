####################################################################
# Hi Robot Configuration
####################################################################
hi_cfg = {
    'joints_name': [
        'A_Waist',
        # Left Arm
        'Shoulder_Y_L',
        'Shoulder_X_L',
        'Shoulder_Z_L',
        'Elbow_L',
        'Wrist_Z_L',
        'Wrist_Y_L',
        'Wrist_X_L',
        # Right Arm
        'Shoulder_Y_R',
        'Shoulder_X_R',
        'Shoulder_Z_R',
        'Elbow_R',
        'Wrist_Z_R',
        'Wrist_Y_R',
        'Wrist_X_R',
        # Left Leg
        'Hip_Z_L',
        'Hip_X_L',
        'Hip_Y_L',
        'Knee_L',
        'Ankle_Y_L',
        'Ankle_X_L',
        # Right Leg
        'Hip_Z_R',
        'Hip_X_R',
        'Hip_Y_R',
        'Knee_R',
        'Ankle_Y_R',
        'Ankle_X_R',
    ],
    'alias': [
        'waist_joint',
        # Left Arm
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_yaw_joint',
        'left_wrist_pitch_joint',
        'left_wrist_roll_joint',
        # Right Arm
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
        'right_wrist_yaw_joint',
        'right_wrist_pitch_joint',
        'right_wrist_roll_joint',
        # Left Leg
        'left_hip_yaw_joint',
        'left_hip_roll_joint',
        'left_hip_pitch_joint',
        'left_knee_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        # Right Leg
        'right_hip_yaw_joint',
        'right_hip_roll_joint',
        'right_hip_pitch_joint',
        'right_knee_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
    ],
    'edges': [
        # ['A_Waist', 'Shoulder_Y_L'],
        # Left Arm
        ['Shoulder_Y_L', 'Shoulder_X_L'],
        ['Shoulder_X_L', 'Shoulder_Z_L'],
        ['Shoulder_Z_L', 'Elbow_L'],
        ['Elbow_L', 'Wrist_Z_L'],
        ['Wrist_Z_L', 'Wrist_Y_L'],
        ['Wrist_Y_L', 'Wrist_X_L'],
        # Right Arm
        # ['A_Waist', 'Shoulder_Y_R'],
        ['Shoulder_Y_R', 'Shoulder_X_R'],
        ['Shoulder_X_R', 'Shoulder_Z_R'],
        ['Shoulder_Z_R', 'Elbow_R'],
        ['Elbow_R', 'Wrist_Z_R'],
        ['Wrist_Z_R', 'Wrist_Y_R'],
        ['Wrist_Y_R', 'Wrist_X_R'],
        # Left Leg
        ['A_Waist', 'Hip_Z_L'],
        ['Hip_Z_L', 'Hip_X_L'],
        ['Hip_X_L', 'Hip_Y_L'],
        ['Hip_Y_L', 'Knee_L'],
        ['Knee_L', 'Ankle_Y_L'],
        ['Ankle_Y_L', 'Ankle_X_L'],
        # Right Leg
        ['A_Waist', 'Hip_Z_R'],
        ['Hip_Z_R', 'Hip_X_R'],
        ['Hip_X_R', 'Hip_Y_R'],
        ['Hip_Y_R', 'Knee_R'],
        ['Knee_R', 'Ankle_Y_R'],
        ['Ankle_Y_R', 'Ankle_X_R'],
    ],
    'root_name': [
        'A_Waist',
        'Shoulder_Y_L',
        'Shoulder_Y_R',
    ],
    'end_effectors': [
        'Wrist_Z_L',
        'Wrist_Z_R',
        'Ankle_Y_L',
        'Ankle_Y_R',
    ],
    'shoulders': [
        'Shoulder_Y_L',
        'Shoulder_Y_R',
        'Hip_Y_L',
        'Hip_Y_R',
    ],
    'elbows': [
        'Elbow_L',
        'Elbow_R',
        'Knee_L',
        'Knee_R',
    ],
    'lower_joints': [
        # Left Leg
        'Hip_Z_L',
        'Hip_X_L',
        'Hip_Y_L',
        'Knee_L',
        'Ankle_Y_L',
        'Ankle_X_L',
        # Right Leg
        'Hip_Z_R',
        'Hip_X_R',
        'Hip_Y_R',
        'Knee_R',
        'Ankle_Y_R',
        'Ankle_X_R',
    ],
    'init_angle': [
        0,   #'A_Waist',
        # Left Arm
        0,   #'Shoulder_Y_L',
        0.4, #'Shoulder_X_L',
        0,   #'Shoulder_Z_L',
        -0.3,#'Elbow_L',
        0,   #'Wrist_Z_L',
        0,   #'Wrist_Y_L',
        0,   #'Wrist_X_L',
        # Right Arm
        0,   #'Shoulder_Y_R',
        -0.4,#'Shoulder_X_R',
        0,   #'Shoulder_Z_R',
        -0.3,#'Elbow_R',
        0,   #'Wrist_Z_R',
        0,   #'Wrist_Y_R',
        0,   #'Wrist_X_R',
        # Left Leg
        0,   #'Hip_Z_L',
        0,   #'Hip_X_L',
        -0.3,#'Hip_Y_L',
        0.5, #'Knee_L',
        0,   #'Ankle_Y_L',
        0,   #'Ankle_X_L',
        # Right Leg
        0,   #'Hip_Z_R',
        0,   #'Hip_X_R',
        -0.3,#'Hip_Y_R',
        0.5, #'Knee_R',
        0,   #'Ankle_Y_R',
        0,   #'Ankle_X_R',
    ],
}

joint2alias = {joint: alias for joint, alias in zip(hi_cfg['joints_name'], hi_cfg['alias'])}

selected_joints = [
    # 'A_Waist',
    # Left Arm
    'Shoulder_Y_L',
    'Shoulder_X_L',
    'Shoulder_Z_L',
    'Elbow_L',
    # 'Wrist_Z_L',
    # 'Wrist_Y_L',
    # 'Wrist_X_L',
    # Right Arm
    'Shoulder_Y_R',
    'Shoulder_X_R',
    'Shoulder_Z_R',
    'Elbow_R',
    # 'Wrist_Z_R',
    # 'Wrist_Y_R',
    # 'Wrist_X_R',
    # Left Leg
    'Hip_Z_L',
    # 'Hip_X_L',
    'Hip_Y_L',
    'Knee_L',
    'Ankle_Y_L',
    # 'Ankle_X_L',
    # Right Leg
    'Hip_Z_R',
    # 'Hip_X_R',
    'Hip_Y_R',
    'Knee_R',
    'Ankle_Y_R',
    # 'Ankle_X_R',
]
selected_joints_alias = [joint2alias[j] for j in selected_joints]

selected_keypos = [
    'A_Waist',
    # Left Arm
    # 'Shoulder_Y_L',
    # 'Shoulder_X_L',
    # 'Shoulder_Z_L',
    'Elbow_L',
    # 'Wrist_Z_L',
    'Wrist_Y_L',
    # 'Wrist_X_L',
    # Right Arm
    # 'Shoulder_Y_R',
    # 'Shoulder_X_R',
    # 'Shoulder_Z_R',
    'Elbow_R',
    # 'Wrist_Z_R',
    'Wrist_Y_R',
    # 'Wrist_X_R',
    # Left Leg
    # 'Hip_Z_L',
    # 'Hip_X_L',
    'Hip_Y_L',
    'Knee_L',
    'Ankle_Y_L',
    # 'Ankle_X_L',
    # Right Leg
    # 'Hip_Z_R',
    # 'Hip_X_R',
    'Hip_Y_R',
    'Knee_R',
    'Ankle_Y_R',
    # 'Ankle_X_R',
]
selected_keypos_alias = [joint2alias[j] for j in selected_keypos]
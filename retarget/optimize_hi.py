import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from urdf2graph import yumi2graph
from kinematics import ForwardKinematicsAxis
from rotations import matrix_to_quaternion, euler_angles_to_matrix
from utils.parsers import BVHParser
from utils.preprocessing import *
from hi_config import hi_cfg

####################################################################
# Mocap
####################################################################
SAVE_ANIMATION = False

class DataType(Enum):
    PFNN = 1
    LAFAN = 2
    XSENS = 3
    CMU = 4

pfnn_mocaps = [
    # '../data/LocomotionFlat01_000_XYZ.bvh',
    # '../data/LocomotionFlat01_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat02_000_XYZ.bvh',
    # '../data/LocomotionFlat02_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat02_001_XYZ.bvh',
    # '../data/LocomotionFlat02_001_mirror_XYZ.bvh',
    # '../data/LocomotionFlat03_000_XYZ.bvh',
    # '../data/LocomotionFlat03_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat04_000_XYZ.bvh',
    # '../data/LocomotionFlat04_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat05_000_XYZ.bvh',
    # '../data/LocomotionFlat05_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat06_000_XYZ.bvh',
    # '../data/LocomotionFlat06_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat06_001_XYZ.bvh',
    # '../data/LocomotionFlat06_001_mirror_XYZ.bvh',
    # '../data/LocomotionFlat07_000_XYZ.bvh',
    # '../data/LocomotionFlat07_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat08_000_XYZ.bvh',
    # '../data/LocomotionFlat08_000_mirror_XYZ.bvh',
    # '../data/LocomotionFlat08_001_XYZ.bvh',
    # '../data/LocomotionFlat08_001_mirror_XYZ.bvh',
]
lafan_mocaps = [
    # '../data/walk1_subject5_XYZ.bvh',
]
cmu_mocaps = [
    # '../data/cmu-mocap/data/009/09_01_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_01_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_02_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_02_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_03_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_03_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_04_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_04_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_05_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_05_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_06_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_06_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_07_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_07_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_08_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_08_mirror_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_09_XYZ.bvh',
    # '../data/cmu-mocap/data/009/09_09_mirror_XYZ.bvh',
]
xsens_mocaps = [
    '../data/xsens-mocap-run/Run-001_XYZ.bvh',
    '../data/xsens-mocap-run/Run-001_mirror_XYZ.bvh',
]
mocap_files = pfnn_mocaps + lafan_mocaps + xsens_mocaps + cmu_mocaps
# data_dir = '../data'
# mocap_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_XYZ.bvh')])
# print(mocap_files)

def mocap_pos(jname):
    jpos = np.stack([positions[jname+'_Xposition'],
                    positions[jname+'_Yposition'],
                    positions[jname+'_Zposition']], axis=-1)
    jpos = torch.from_numpy(jpos)
    return jpos

def mocap_rot(jname):
    jrot = R.from_euler('XYZ', np.stack([rotations[jname+'_Xrotation'],
                                        rotations[jname+'_Yrotation'],
                                        rotations[jname+'_Zrotation']], axis=-1), degrees=True)
    jrot = torch.from_numpy((jrot * R.from_euler('z', -90, degrees=True)).as_euler('XYZ')).float()
    return jrot

def get_foot_names():
    if DATA_TYPE == DataType.PFNN:
        foot_names = ['LeftFoot', 'RightFoot']
    elif DATA_TYPE == DataType.LAFAN:
        foot_names = ['lAnkle', 'rAnkle']
    elif DATA_TYPE == DataType.XSENS:
        foot_names = ['LeftAnkle', 'RightAnkle']
    elif DATA_TYPE == DataType.CMU:
        foot_names = ['LeftToeBase', 'RightToeBase']
    else:
        raise Exception("Error Data Type")
    return foot_names

def force_foot_contact(positions):
    foot_names = get_foot_names()
    foot_pos = torch.stack([mocap_pos(foot) for foot in foot_names], dim=1)
    min_z = torch.min(foot_pos[..., 2], dim=1)[0].numpy()
    for name in positions.columns:
        if '_Zposition' in name:
            positions[name] -= min_z

def mocap_contact():
    # source contact
    foot_names = get_foot_names()
    foot_pos = torch.stack([mocap_pos(foot) for foot in foot_names], dim=1)
    if DATA_TYPE == DataType.PFNN:
        foot_contact = torch.logical_and(foot_pos[1:,:,2] <= 3.0,
            torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1) / new_framerate <= 3)  # [T-1, 2]
    elif DATA_TYPE == DataType.LAFAN:
        foot_contact = torch.logical_and(foot_pos[1:,:,2] <= 0.08,
            torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1) / new_framerate <= 3 / 100.)  # [T-1, 2]
    elif DATA_TYPE == DataType.XSENS:
        foot_contact = torch.logical_and(foot_pos[1:,:,2] <= 15.0,
            torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1) / new_framerate <= 15.0)  # [T-1, 2]
    elif DATA_TYPE == DataType.CMU:
        foot_contact = torch.logical_and(foot_pos[1:,:,2] <= 3.0,
            torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1) / new_framerate <= 5.0)  # [T-1, 2]
    else:
        raise Exception("Error Data Type")
    # print(foot_pos.shape, foot_contact.shape)
    # print(foot_contact, foot_pos[1:,:,2], torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1) / new_framerate)
    return foot_contact

def constrain_ang(x):
    lower = hi.lower.squeeze().unsqueeze(0)
    upper = hi.upper.squeeze().unsqueeze(0)
    return lower + (upper - lower)*x.sigmoid()

def ang2x(ang):
    lower = hi.lower.squeeze().unsqueeze(0)
    upper = hi.upper.squeeze().unsqueeze(0)
    sig = (ang - lower) / (upper - lower)
    x = torch.log(sig / (1 - sig))
    return x

def scale_root_pos(root_pos):
    delta = root_pos[1:] - root_pos[:-1]
    delta *= hi_leg / mocap_leg
    new_root_pos = torch.cat((hi_init_pos.unsqueeze(0), hi_init_pos + torch.cumsum(delta, dim=0)), dim=0)
    # new_root_pos = []
    # for t in range(T):
    #     if t == 0: root_pos_t = hi_init_pos
    #     else: root_pos_t += delta[t-1]
    #     new_root_pos.append(root_pos_t.clone())
    # new_root_pos = torch.stack(new_root_pos)
    return new_root_pos

for mocap_file in mocap_files:
    DATA_TYPE = None
    if mocap_file in pfnn_mocaps:
        DATA_TYPE = DataType.PFNN
    elif mocap_file in lafan_mocaps:
        DATA_TYPE = DataType.LAFAN
    elif mocap_file in xsens_mocaps:
        DATA_TYPE = DataType.XSENS
    elif mocap_file in cmu_mocaps:
        DATA_TYPE = DataType.CMU
    else:
        raise Exception("Error mocap file")
    print(DATA_TYPE, mocap_file)
    # bvh parser
    bvh_parser = BVHParser()
    parsed_data = bvh_parser.parse(mocap_file)
    # Downsample the framerate
    new_framerate = 1./50.
    step_size = round(new_framerate / parsed_data.framerate)
    # Force acceleration
    # step_size = 8
    parsed_data.values = parsed_data.values.iloc[::step_size]
    parsed_data.framerate = new_framerate
    # Convert to position
    mp = MocapParameterizer('position')
    rotations = parsed_data.values
    positions = mp.fit_transform([parsed_data])[0].values
    T = positions.shape[0]
    # T pose
    tpose_data = parsed_data.clone()
    for col in tpose_data.values.columns:
        tpose_data.values[col].values[:] = 0
    tpose_positions = mp.fit_transform([tpose_data])[0].values
    # mocap_leg = (tpose_positions['Hips_Zposition'] - tpose_positions['LeftFoot_Zposition']).to_numpy()[0]
    if DATA_TYPE == DataType.PFNN or DATA_TYPE == DataType.CMU:
        mocap_leg = (tpose_positions['LeftUpLeg_Zposition'] - tpose_positions['LeftFoot_Zposition']).to_numpy()[0]
    elif DATA_TYPE == DataType.LAFAN:
        mocap_leg = (tpose_positions['lHip_Zposition'] - tpose_positions['lToeJoint_Zposition']).to_numpy()[0]
    elif DATA_TYPE == DataType.XSENS:
        mocap_leg = (tpose_positions['Hips_Zposition'] - tpose_positions['LeftToe_Zposition']).to_numpy()[0]
    else:
        raise Exception("Error Data Type")

    contact = mocap_contact()  # [T-1, 2]

    # Hi graph
    hi = yumi2graph('./Hi.urdf', hi_cfg)
    hi_leg = 0.9
    hi_init_pos = torch.tensor([0, 0, 1.02])
    lower_indices = [hi_cfg['joints_name'].index(joint) for joint in hi_cfg['lower_joints']]

    # optimization variables
    data = Batch.from_data_list([hi]*T)
    root_pos = scale_root_pos(mocap_pos(parsed_data.root_name))
    root_vel = (root_pos[1:] - root_pos[:-1]) / parsed_data.framerate
    root_rot = mocap_rot(parsed_data.root_name)
    root_rot[:, :2] = 0
    x = data.x.reshape(T, -1)
    # Initialize x
    init_angle = torch.tensor(hi_cfg['init_angle']).unsqueeze(0).repeat(T, 1)
    x = ang2x(init_angle)
    x.requires_grad = True
    root_vel.requires_grad = True
    # root_rot.requires_grad = True
    # optimizer
    optimizer = optim.Adam([{"params": root_vel, "lr": 1e-2},
                            # {"params": root_rot, "lr": 5e-3},
                            {"params": x, "lr": 5e-2}],
                            lr=1e-3, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    fk = ForwardKinematicsAxis()

    # start optimizing
    for iter in tqdm(range(1000)):
        # zero gradient
        optimizer.zero_grad()
        # detach upper body joints
        detach_x = []
        for idx, joint in enumerate(hi_cfg['joints_name']):
            if joint in ['Base', 'A_Waist', 'Wrist_Z_L', 'Wrist_Y_L', 'Wrist_X_L', 'Wrist_Z_R', 'Wrist_Y_R', 'Wrist_X_R', 'Hip_X_L', 'Hip_X_R']:
                detach_x.append(x[:, idx].detach())
            else:
                detach_x.append(x[:, idx])
        detach_x = torch.stack(detach_x, dim=1)
        ang = constrain_ang(detach_x)
        root_pos = torch.cat((hi_init_pos.unsqueeze(0), hi_init_pos + torch.cumsum(root_vel, dim=0) * parsed_data.framerate), dim=0)
        # detach_root_pos = torch.cat((root_pos[:, 0:2].detach(), root_pos[:, 2:3]), dim=-1)
        # forward kinematics
        local_pos, local_rot, _ = fk(ang, data.parent, data.offset, data.num_graphs, data.axis)
        root_R = euler_angles_to_matrix(root_rot, 'XYZ').float()
        local_pos = torch.matmul(root_R, local_pos.reshape(T, -1, 3).permute(0, 2, 1)).permute(0, 2, 1)
        global_pos = local_pos.detach() + root_pos.unsqueeze(1)
        global_rot = torch.matmul(root_R.reshape(T, -1, 3, 3), local_rot.reshape(T, -1, 3, 3))
        # vector loss
        if DATA_TYPE == DataType.PFNN or DATA_TYPE == DataType.CMU:
            source_sh = torch.stack([mocap_pos('LeftArm'), mocap_pos('RightArm'), mocap_pos('LeftUpLeg'), mocap_pos('RightUpLeg')], dim=1).float().detach()
            source_el = torch.stack([mocap_pos('LeftForeArm'), mocap_pos('RightForeArm'), mocap_pos('LeftLeg'), mocap_pos('RightLeg')], dim=1).float().detach()
            source_ee = torch.stack([mocap_pos('LeftHand'), mocap_pos('RightHand'), mocap_pos('LeftFoot'), mocap_pos('RightFoot')], dim=1).float().detach()
        elif DATA_TYPE == DataType.LAFAN:
            source_sh = torch.stack([mocap_pos('lShoulder'), mocap_pos('rShoulder'), mocap_pos('lHip'), mocap_pos('rHip')], dim=1).float().detach()
            source_el = torch.stack([mocap_pos('lElbow'), mocap_pos('rElbow'), mocap_pos('lKnee'), mocap_pos('rKnee')], dim=1).float().detach()
            source_ee = torch.stack([mocap_pos('lWrist'), mocap_pos('rWrist'), mocap_pos('lAnkle'), mocap_pos('rAnkle')], dim=1).float().detach()
        elif DATA_TYPE == DataType.XSENS:
            source_sh = torch.stack([mocap_pos('LeftShoulder'), mocap_pos('RightShoulder'), mocap_pos('LeftHip'), mocap_pos('RightHip')], dim=1).float().detach()
            source_el = torch.stack([mocap_pos('LeftElbow'), mocap_pos('RightElbow'), mocap_pos('LeftKnee'), mocap_pos('RightKnee')], dim=1).float().detach()
            source_ee = torch.stack([mocap_pos('LeftWrist'), mocap_pos('RightWrist'), mocap_pos('LeftAnkle'), mocap_pos('RightAnkle')], dim=1).float().detach()
        else:
            raise Exception("Error Data Type")
        target_sh = local_pos[:, [hi_cfg['joints_name'].index(j) for j in hi_cfg['shoulders']], :]
        target_el = local_pos[:, [hi_cfg['joints_name'].index(j) for j in hi_cfg['elbows']], :]
        target_ee = local_pos[:, [hi_cfg['joints_name'].index(j) for j in hi_cfg['end_effectors']], :]
        # print(target_sh.shape, target_el.shape, target_ee.shape, source_sh.shape, source_el.shape, source_ee.shape)
        target_vector1 = F.normalize(target_el - target_sh, dim=-1)
        target_vector2 = F.normalize(target_ee - target_el, dim=-1)
        source_vector1 = F.normalize(source_el - source_sh, dim=-1)
        source_vector2 = F.normalize(source_ee - source_el, dim=-1)
        # print(target_vector1.shape, target_vector2.shape, source_vector1.shape, source_vector2.shape)
        vector1_loss = criterion(target_vector1, source_vector1)
        vector2_loss = criterion(target_vector2, source_vector2)
        vec_loss = (vector1_loss + vector2_loss)
        # print(normalize_source_vector1[0], normalize_target_vector1[0])
        # print(normalize_source_vector2[0], normalize_target_vector2[0])
        # foot orientation
        lfoot_rot = global_rot[:, hi_cfg['joints_name'].index('Ankle_X_L')]
        lfoot_rot = matrix_to_quaternion(lfoot_rot)
        rfoot_rot = global_rot[:, hi_cfg['joints_name'].index('Ankle_X_R')]
        rfoot_rot = matrix_to_quaternion(rfoot_rot)
        foot_loss = ((lfoot_rot[:, 1:3])**2).mean() + ((rfoot_rot[:, 1:3])**2).mean()
        # foot contact
        foot_pos = global_pos[:, [hi_cfg['joints_name'].index(j) for j in ['Ankle_X_L', 'Ankle_X_R']], :]  # [T, 2, 3]
        foot_height = foot_pos[1:,:,2]  # [T-1, 2]
        contact_height = 1000*((torch.masked_select(foot_height, contact) - 0.09)**2).mean()
        foot_vel = torch.norm(foot_pos[1:,:,:]-foot_pos[:-1,:,:], dim=-1)  # [T-1, 2]
        contact_vel = 1000*(torch.masked_select(foot_vel, contact)**2).mean()
        min_Z = torch.min(global_pos[..., 2], dim=-1, keepdims=True).values
        # force_contact = 10000*((min_Z - 0.035)**2).mean()
        contact_loss = contact_height + contact_vel
        # contact_loss = force_contact
        # smoothness
        # root_vel = root_pos[1:] - root_pos[:-1]
        root_acc = root_vel[1:] - root_vel[:-1]
        ang_vel = ang[1:] - ang[:-1]
        ang_acc = ang_vel[1:] - ang_vel[:-1]
        smooth_loss = 100*torch.norm(root_acc, dim=-1).mean() + 1*torch.norm(ang_acc[:, lower_indices], dim=-1).mean()

        # loss = vec_loss + foot_loss
        loss = vec_loss + foot_loss + contact_loss #+ smooth_loss
        print(loss.item(), vec_loss.item(), foot_loss.item(), contact_loss.item(), smooth_loss.item())
        # backward
        loss.backward()
        # optimize
        optimizer.step()

    vel = (ang[1:] - ang[:-1]) / parsed_data.framerate
    vel = torch.cat([vel[:1], vel], dim=0)
    contact = torch.cat([contact[:1], contact], dim=0)

    # Interpolation
    from scipy.interpolate import CubicSpline
    # spline fitting
    t = np.arange(T) * parsed_data.framerate
    cs = CubicSpline(t[::4], ang[::4].detach().cpu().numpy())
    smooth_ang = cs(t)
    # smooth_vel = cs(t, 1)
    # smooth_acc = cs(t, 2)

    joints = hi_cfg['joints_name']
    jidx = [hi_cfg['joints_name'].index(j) for j in joints]

    hf = h5py.File(os.path.join('./saved/hi/', mocap_file.split('/')[-1][:-4] + '.h5'), 'w')
    g1 = hf.create_group('group1')
    g1.create_dataset('joint_angle', data=ang[:,jidx].detach().cpu().numpy())
    g1.create_dataset('smooth_ang', data=smooth_ang[:,jidx])
    g1.create_dataset('joint_vel', data=vel[:,jidx].detach().cpu().numpy())
    g1.create_dataset('joint_pos', data=global_pos[:,jidx].detach().cpu().numpy())
    # g1.create_dataset('local_rot', data=R.from_matrix(local_rot.detach().cpu().numpy()).as_quat().reshape(T,-1,4)[:,jidx])
    g1.create_dataset('local_rot', data=local_rot.reshape(T, -1, 3, 3)[:,jidx].detach().cpu().numpy())
    g1.create_dataset('global_rot', data=global_rot[:,jidx].detach().cpu().numpy())
    # g1.create_dataset('global_rot', data=R.from_matrix(global_rot[:,jidx].detach().cpu().numpy().reshape(-1,3,3)).as_quat().reshape(T,-1,4))
    g1.create_dataset('root_pos', data=root_pos.detach().cpu().numpy())
    # g1.create_dataset('root_rot', data=R.from_euler('XYZ', root_rot.detach().cpu().numpy()).as_quat())
    g1.create_dataset('root_rot', data=root_rot.detach().cpu().numpy())
    g1.create_dataset('contact', data=contact.detach().cpu().numpy())
    hf.close()
    print('Target H5 file saved!')

    ###################################################################
    # AMP format
    ###################################################################
    # amp_obs_data = np.concatenate((
    #     root_pos[2:, :].detach().cpu().numpy(),
    #     root_rot_quat[2:, :],
    #     ang[2: jidx].detach().cpu().numpy(),
    #     local_root_vel[2:, :].detach().cpu().numpy(),
    #     local_ang_vel[2:, :].detach().cpu().numpy(),
    #     local_pos.reshape(T, -1, 3)[2:, foot_idx_L].detach().cpu().numpy(),
    #     local_pos.reshape(T, -1, 3)[2:, foot_idx_R].detach().cpu().numpy(),
    #     vel[2:, jidx].detach().cpu().numpy(),
    # ), axis=1)
    # print(amp_obs_data.shape)

    # with open(os.path.join('./saved/hi/',  mocap_file.split('/')[-1][:-4] + '.txt'), 'w') as f:
    #     f.write('{"LoopMode": "Wrap",\n"FrameDuration": 0.0083,\n"EnableCycleOffsetPosition": true,\n"EnableCycleOffsetRotation": true,\n"MotionWeight": 1.0,\n"Frames":[')
    #     for i in range(len(amp_obs_data)):
    #         f.write('[')
    #         for j in range(len(amp_obs_data[0])):
    #             if j == len(amp_obs_data[0]) - 1:
    #                 f.write(str(amp_obs_data[i,j]))
    #             else:
    #                 f.write(str(amp_obs_data[i,j])+',')
    #         if i == len(amp_obs_data) - 1:
    #             f.write(']'+'\n')
    #         else:
    #             f.write(']'+','+'\n')
    #     f.write(']}')
    # print('Target txt file saved!')

    ###################################################################
    # Animation
    ###################################################################
    pos = global_pos.detach().cpu().numpy()
    if SAVE_ANIMATION:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        def run(t):
            ## left
            Pos_X = pos[t, :, 0]
            Pos_Y = pos[t, :, 1]
            Pos_Z = pos[t, :, 2]
            for line, edge in zip(lines, hi_cfg['edges']):
                joint_a = hi_cfg['joints_name'].index(edge[0])
                joint_b = hi_cfg['joints_name'].index(edge[1])
                line_x = [Pos_X[joint_a], Pos_X[joint_b]]
                line_y = [Pos_Y[joint_a], Pos_Y[joint_b]]
                line_z = [Pos_Z[joint_a], Pos_Z[joint_b]]
                line.set_data(np.array([line_x, line_y]))
                line.set_3d_properties(np.array(line_z))
            return lines

        # attach 3D axis to figure
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ## left
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.add_axes(ax)
        ax.set_title('Source')
        ax.view_init(elev=30, azim=45)
        ax.set_xlim3d([-3.0, 3.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.0, 6.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 6.0])
        ax.set_zlabel('Z')
        lines = [ax.plot([], [], [], 'royalblue', marker='o')[0] for i in range(len(hi_cfg['edges']))]
        # create animation
        ani = animation.FuncAnimation(fig, run, np.arange(T), interval=50)
        FFwriter = animation.FFMpegWriter(fps=30)
        plt.show()
        ani.save(os.path.join('./saved/hi/',  mocap_file.split('/')[-1][:-4] + '.mp4'), writer=FFwriter)
        print('Animation saved!')
import os
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class HiEnv(gym.Env):
    """docstring for HiEnv"""
    def __init__(self, viz_ee_ori=False):
        super(HiEnv, self).__init__()
        self.pos = np.array([0,0,1.02])
        self.rot = np.array([0,0,0,1])
        self.ctrl_joints = [
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
        ]
        self.viz_ee_ori = viz_ee_ori
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-10, cameraTargetPosition=[0,0,0.8])
        self.reset()
    
    def update_root(self, root_pos, root_rot):
        self.pos = root_pos
        self.rot = root_rot

    def step(self, action, custom_reward=None):
        base_local_inertial_pos = p.getDynamicsInfo(self.Uid,-1)[3]
        base_local_inertial_orn = p.getDynamicsInfo(self.Uid,-1)[4]
        new_pos, new_rot = p.multiplyTransforms(self.pos, self.rot, base_local_inertial_pos, base_local_inertial_orn)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.resetBasePositionAndOrientation(self.Uid, new_pos, new_rot)
        # self.pos = self.pos + np.array([0, 0, 0.01])
        # print(self.pos)

        # p.setJointMotorControlArray(self.Uid, [self.joint2Index[joint] for joint in self.ctrl_joints], p.POSITION_CONTROL, action)
        # p.stepSimulation()
        for jidx, joint in enumerate(self.ctrl_joints):
            p.resetJointState(self.Uid, self.joint2Index[joint], action[jidx])
        # get states
        jointStates = {}
        for joint in self.ctrl_joints:
            jointStates[joint] = p.getJointState(self.Uid, self.joint2Index[joint])
        linkStates = {}
        for link in self.link_names:
            linkStates[link] = p.getLinkState(self.Uid, self.link2Index[link])
        # recover color
        for index, color in self.linkColor.items():
            p.changeVisualShape(self.Uid, index, rgbaColor=color)
        # check collision
        collision = False
        # for link in self.link_names:
        #     if len(p.getContactPoints(bodyA=self.Uid, linkIndexA=self.link2Index[link])) > 0:
        #         collision = True
        #         for contact in p.getContactPoints(bodyA=self.Uid, bodyB=self.Uid, linkIndexA=self.link2Index[link]):
        #             print("Collision Occurred in Link {} & Link {}!!!".format(contact[3], contact[4]))
        #             p.changeVisualShape(self.Uid, contact[3], rgbaColor=[1,0,0,1])
        #             p.changeVisualShape(self.Uid, contact[4], rgbaColor=[1,0,0,1])
        # Plot End Effector Orientation
        if self.viz_ee_ori:
            p.removeAllUserDebugItems()
            for name in ['HAND_L','HAND_R', 'FOOT_L', 'FOOT_R']:
                worldLinkFramePosition, worldLinkFrameOrientation = p.getLinkState(self.Uid, self.link2Index[name])[4:6]
                linkRotMat = np.array(p.getMatrixFromQuaternion(worldLinkFrameOrientation)).reshape(3,3)
                x_axis = linkRotMat[:,0]
                x_end = (np.array(worldLinkFramePosition) + np.array(x_axis)*0.2).tolist()
                self.x_line_id = p.addUserDebugLine(worldLinkFramePosition, x_end, (1,0,0))
                y_axis = linkRotMat[:,1]
                y_end = (np.array(worldLinkFramePosition) + np.array(y_axis)*0.2).tolist()
                self.y_line_id = p.addUserDebugLine(worldLinkFramePosition, y_end, (0,1,0))
                z_axis = linkRotMat[:,2]
                z_end = (np.array(worldLinkFramePosition) + np.array(z_axis)*0.2).tolist()
                self.z_line_id = p.addUserDebugLine(worldLinkFramePosition, z_end, (0,0,1))
        
        self.step_counter += 1

        if custom_reward is None:
            # default reward
            reward = 0
            done = False
        else:
            # custom reward
            reward, done = custom_reward(jointStates=jointStates, linkStates=linkStates, collision=collision, step_counter=self.step_counter)

        info = {'collision': collision}
        observation = [jointStates[joint][0] for joint in self.ctrl_joints]
        return observation, reward, done, info

    def reset(self):
        p.resetSimulation()
        self.step_counter = 0
        self.Uid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            "assets/Hi_imu/urdf/Hi.urdf"), basePosition=self.pos, baseOrientation=self.rot, useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadMJCF("mjcf/ground_plane.xml") # ground plane
        p.setGravity(0,0,-10)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./240.)
        self.joint2Index = {} # index map to jointName
        self.link_names = []
        self.link2Index = {} # index map to linkName
        self.lower_limits = np.zeros(len(self.ctrl_joints))
        self.upper_limits = np.zeros(len(self.ctrl_joints))
        self.init_angles = np.zeros(len(self.ctrl_joints))
        for index in range(p.getNumJoints(self.Uid)):
            jointName = p.getJointInfo(self.Uid, index)[1].decode('utf-8')
            linkName = p.getJointInfo(self.Uid, index)[12].decode('utf-8')
            if jointName in self.ctrl_joints:
                self.joint2Index[jointName] = index
                self.link_names.append(linkName)
                self.link2Index[linkName] = index
                self.lower_limits[self.ctrl_joints.index(jointName)] = p.getJointInfo(self.Uid, index)[8]
                self.upper_limits[self.ctrl_joints.index(jointName)] = p.getJointInfo(self.Uid, index)[9]
        self.linkColor = {} # index map to jointColor
        for data in p.getVisualShapeData(self.Uid):
            linkIndex, rgbaColor = data[1], data[7]
            self.linkColor[linkIndex] = rgbaColor
        self.action_space = spaces.Box(np.array([-1]*len(self.ctrl_joints)), np.array([1]*len(self.ctrl_joints)))
        self.observation_space = spaces.Box(np.array([-1]*len(self.ctrl_joints)), np.array([1]*len(self.ctrl_joints)))

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,0,0.5],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960)/720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960,4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()

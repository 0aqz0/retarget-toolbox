import gym, humanoid_gym
import pybullet as p
import numpy as np
import h5py
import argparse
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Play H5 Motion File")
parser.add_argument('--env_name', type=str, default='hi-v0', help="Environment Name (hi-v0, wukong-v0, g1-v0)")
parser.add_argument('--h5_path', type=str, default='../retarget/saved/hi/Walk-001_XYZ.h5', help="H5 File Path")
args = parser.parse_args()


hf = h5py.File(args.h5_path, 'r')
group1 = hf.get('group1')
joint_angles = group1.get('joint_angle')
joint_pos = group1.get('joint_pos')
root_pos = group1.get('root_pos')
root_rot = R.from_euler('XYZ', group1.get('root_rot')).as_quat()
contact = group1.get('contact')
total_frames = joint_angles.shape[0]
print(joint_angles.shape)


env = gym.make(args.env_name)
env.reset()
env.render()
# observation = env.reset()

while True:
    env.render()

    for t in range(total_frames):
        env.update_root(root_pos[t], root_rot[t])
        action = joint_angles[t].tolist()
        if t > 0:
            p.removeAllUserDebugItems()
            p.addUserDebugText('contact state: {} {}'.format(contact[t-1, 0], contact[t-1, 1]), (0,0,1.5), textSize=5)
        # print(action)
        observation, reward, done, info = env.step(action)

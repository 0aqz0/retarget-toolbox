import gym, humanoid_gym
import pybullet as p
import numpy as np
import h5py


file = './saved/wukong-v1-2/Take 2023-12-17 05.08.16 PM_CZH_XYZ.h5'
hf = h5py.File(file, 'r')
group1 = hf.get('group1')
joint_angles = group1.get('joint_angle')
# joint_pos = group1.get('joint_pos')
root_pos = group1.get('root_pos')
root_rot = group1.get('root_rot')
contact = group1.get('contact')
# forces = group1.get('contact_force')
# print(forces.shape)
total_frames = joint_angles.shape[0]
print(joint_angles.shape)


env = gym.make('wukong-v0')
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
        for _ in range(2):
            observation, reward, done, info = env.step(action)

import gym, humanoid_gym
import pybullet as p
import argparse

parser = argparse.ArgumentParser(description="Human Control Interface")
parser.add_argument('--env_name', type=str, default='hi-v0', help="Environment Name (hi-v0, wukong-v0, g1-v0)")
args = parser.parse_args()

env = gym.make(args.env_name)
env.reset()

motorIds = []
# robot joints
for name, lower, upper, init in zip(env.ctrl_joints, env.lower_limits, env.upper_limits, env.init_angles):
    motorIds.append(p.addUserDebugParameter(name, lower, upper, init))

while True:
    env.render()

    actions = []
    for motorId in motorIds:
        actions.append(p.readUserDebugParameter(motorId))

    observation, reward, done, info = env.step(actions)

from gym.envs.registration import register

# WuKongIV
register(
    id='wukong-v0',
    entry_point='humanoid_gym.envs:WuKongEnv',
)

# Hi
register(
    id='hi-v0',
    entry_point='humanoid_gym.envs:HiEnv',
)

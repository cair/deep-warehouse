
from gym.envs.registration import register

register(
    id='deep-logistics-normal-v0',
    entry_point='gym_deep_logistics.gym_deep_logistics.envs:DeepLogisticsNormal',
)

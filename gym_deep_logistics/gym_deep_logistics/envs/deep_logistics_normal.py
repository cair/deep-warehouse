import gym
from gym.spaces import Box

from deep_logistics.environment import Environment
from deep_logistics.scheduler import OnDemandScheduler
from deep_logistics.spawn_strategy import LocationSpawnStrategy
from experiments.experiment_3.reward_functions import Reward0
from experiments.experiment_3.state_representations import State1


class DeepLogisticsNormal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.env = Environment(
            height=11,
            width=11,
            depth=3,
            ups=None,
            ticks_per_second=10,
            taxi_n=1,
            taxi_agent=None,
            taxi_respawn=False,
            scheduler=OnDemandScheduler,
            delivery_locations=None,
            spawn_strategy=LocationSpawnStrategy,
            graphics_render=True,
            graphics_tile_height=32,
            graphics_tile_width=32
        )
        self.frame_skip = 4
        self.agent = self.env.get_agent(0)
        self.sgen = State1(self.env)
        self._seed = 0
        self.action_space = self.env.action_space
        self.observation_space = self.sgen.generate(self.agent).shape

    def _step(self, action):

        self.agent.do_action(action)

        #for _ in range(self.frame_skip):  # TODO will fuck up reward
        self.env.update()

        state1 = self.sgen.generate(self.agent)
        reward, terminal = Reward0(self.agent)

        self.env.render()
        return state1, reward, terminal, None

    def _reset(self):
        self.env.reset()
        return self.sgen.generate(self.agent)

    def _render(self, mode='human', close=False):
        return self.sgen.generate(self.agent)

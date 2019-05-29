import gym

from deep_logistics.environment import Environment
from deep_logistics.scheduler import OnDemandScheduler
from deep_logistics.spawn_strategy import LocationSpawnStrategy
from experiments.experiment_3.reward_functions import Reward0
from experiments.experiment_3.state_representations import State1, StateFull


class DeepLogisticsNormal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.env = Environment(
            height=5,
            width=5,
            depth=3,
            ups=None,
            ticks_per_second=1,
            taxi_n=1,
            taxi_agent=None,
            taxi_respawn=False,
            taxi_control="constant",
            scheduler=OnDemandScheduler,
            delivery_locations=None,
            spawn_strategy=LocationSpawnStrategy,
            graphics_render=True,
            graphics_tile_height=16,
            graphics_tile_width=16
        )
        self.frame_skip = 4
        self.agent = self.env.get_agent(0)
        self.sgen = StateFull(self.env)
        self._seed = 0

        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = self.sgen.generate(self.agent).shape

    def step(self, action):

        self.agent.do_action(action)

        #for _ in range(self.frame_skip):  # TODO will fuck up reward
        self.env.update()
        self.env.render()

        state1 = self.sgen.generate(self.agent)
        reward, terminal = Reward0(self.agent)
        if terminal:
            info = dict(
                deliveries=self.agent.total_deliveries,
                pickups=self.agent.total_pickups
            )
        else:
            info = None

        return state1, reward, terminal, info

    def reset(self):
        self.env.reset()
        return self.sgen.generate(self.agent)

    def render(self, mode='human', close=False):
        return self.sgen.generate(self.agent)

import sys
sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")

from agents import AIAgent
import os
from deep_logistics.environment import Environment
from deep_logistics.agent import Agent, ManhattanAgent
from ray.rllib import MultiAgentEnv

from state_representations import State0
from gym.spaces import Tuple, Discrete


class Statistics:

    def __init__(self):
        self.deliveries_before_crash = 1
        self.pickups_before_crash = 1


class DeepLogisticBase(MultiAgentEnv):

    def __init__(self, height, width, ai_count, agent_count, agent, ups, delivery_points, state, render_screen=False):
        self.render_screen = render_screen
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = Environment(
            height=height,
            width=width,
            depth=3,
            agents=ai_count,
            agent_class=agent,
            draw_screen=self.render_screen,
            tile_height=32,
            tile_width=32,
            #scheduler=RandomScheduler,
            ups=ups,
            ticks_per_second=1,
            spawn_interval=1,  # In steps
            task_generate_interval=1,  # In steps
            task_assign_interval=1,  # In steps
            delivery_points=delivery_points
        )

        self.statistics = Statistics()

        assert ai_count < agent_count

        self.state_representation = state(self.env)
        self.observation_space = self.state_representation.generate(self.env.agents[0])
        self.action_space = Discrete(self.env.action_space.N_ACTIONS)

        self.grouping = {'group_1': ["agent_%s" % x for x in range(ai_count)]}
        self.agents = {k: self.env.agents[i] for i, k in enumerate(self.grouping["group_1"])}
        obs_space = Tuple([self.observation_space for _ in range(ai_count)])
        act_space = Tuple([self.action_space for _ in range(ai_count)])

        """self.with_agent_groups(
            groups=self.grouping,
            obs_space=obs_space,
            act_space=act_space
        )"""

        """Spawn all agents etc.."""
        self.env.deploy_agents()
        self.env.task_assignment()

        self.episode = 0

    def get_agents(self):
        return self.env.agents




class DeepLogisticsA10M20x20D4(DeepLogisticBase):

    def __init__(self, args):
        DeepLogisticBase.__init__(self,
                                  height=10,
                                  width=10,
                                  ai_count=1,
                                  agent_count=15,
                                  render_screen=False,
                                  agent=AIAgent,
                                  ups=None,
                                  delivery_points=[
                                      (7, 2),
                                      (2, 2),
                                      (2, 7),
                                      (7, 7)
                                  ],
                                  state=State0)


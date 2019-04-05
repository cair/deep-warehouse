import os
import sys
import time
import glob
from threading import Thread

sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")

import multiprocessing
import argparse
import state_representations
from agent_factory import AgentFactory
from deep_logistics.environment import Environment
from agents import AIAgent
from deep_logistics.agent import Agent
import matplotlib.pyplot as plt


class Env:

    def __init__(self, state_representation, fps=60, ups=None):
        self.env = Environment(
            height=10,
            width=10,
            depth=3,
            agents=1,
            agent_class=AIAgent,
            draw_screen=True,
            tile_height=32,
            tile_width=32,
            # scheduler=RandomScheduler,
            ups=ups,
            ticks_per_second=1,
            spawn_interval=1,  # In seconds
            task_generate_interval=1,  # In seconds
            task_assign_interval=1,  # In seconds
            delivery_points=[
                (7, 2),
                (2, 2),
                (2, 7),
                (7, 7)
            ]
        )

        self.state_representation = state_representation(self.env)

        # Assumes that all agnets have spawned already and that all tasks are assigned.
        self.env.deploy_agents()
        self.env.task_assignment()
        self.last_time = time.time()
        self.pickup_count = 0
        self.delivery_count = 0
        self.stat_deliveries = []
        self.episode = 0

        # env.daemon = True
        # env.start()

        self.player = self.env.agents[0]

    def step(self, action):
        state = self.player.state
        self.player.do_action(action=action)
        self.env.update()
        new_state = self.player.state
        # print("%s => %s" % (state, new_state))

        """Fast-forward the game until the player is respawned."""
        while self.player.state == Agent.INACTIVE:
            self.env.update()

        state = self.state_representation.generate(self.env.agents[0])

        if self.player.state in [Agent.IDLE, Agent.MOVING]:
            reward = -0.01
            terminal = False
        elif self.player.state in [Agent.PICKUP]:
            self.pickup_count += 1
            reward = 1
            terminal = False
            # print("Pickup", state, self.player.task.c_1)
        elif self.player.state in [Agent.DELIVERY]:
            self.delivery_count += 1
            reward = 10
            terminal = False
            # print("Delivery", state)
        elif self.player.state in [Agent.DESTROYED]:
            reward = -1
            terminal = True

        else:
            raise NotImplementedError("Should never happen. all states should be handled somehow")

        return state, reward, terminal, {}

    def reset(self):
        print("[%s] Environment was reset, took: %s seconds. Pickups: %s, Deliveries: %s" % (
        self.episode, time.time() - self.last_time, self.pickup_count, self.delivery_count))
        self.last_time = time.time()
        self.stat_deliveries.append(self.delivery_count)
        if self.episode % 50 == 0:
            self.graph()

        self.pickup_count = 0
        self.delivery_count = 0
        self.episode += 1
        self.env.reset()

    def render(self):
        self.env.render()
        return self.state_representation.generate(self.env.agents[0])

    def graph(self):
        plt.plot([x for x in range(len(self.stat_deliveries))], self.stat_deliveries, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Number of Successive Deliveries')
        plt.title('Deep Logistics - PPO - Experiment A')
        plt.savefig("./ppo-experiment.png")


class TrainWorker(multiprocessing.Process):

    def __init__(self, process_id, agent_type):
        super().__init__()
        self.agent_type = agent_type
        self.process_id = process_id
        self.import_buffer = None
        self.t = None
        self.agent = None

    def experience_import_loop(self):
        while True:
            experience_files = glob.glob('/data/*.npy', recursive=True)
            time.sleep(.1)

    def run(self):
        self.import_buffer = []
        self.t = Thread(target=self.experience_import_loop)
        self.t.start()

        env = Env(state_representation=state_representations.State0)
        self.agent = AgentFactory.create(self.agent_type, env)

        while True:
            # Retrieve the latest (observable) environment state
            state = env.render()  # (float array of shape [10])

            # Query the agent for its action decision
            action = self.agent.act(states=state)  # (scalar between 0 and 4)

            # Execute the decision and retrieve the current performance score
            state_1, reward, terminal, _ = env.step(action)  # (any scalar float)

            if terminal:

                env.reset()

            # Pass feedback about performance (and termination) to the agent
            self.agent.observe(reward=reward, terminal=terminal)


class ExplorationWorker(multiprocessing.Process):

    def __init__(self, process_id, agent_type):
        super().__init__()
        self.process_id = process_id
        self.agent_type = agent_type
        self.agent = None
        self.experiences = None

    def run(self):
        self.experiences = []
        env = Env(state_representation=state_representations.State0)

        self.agent = AgentFactory.create(self.agent_type, env)

        while True:
            # Retrieve the latest (observable) environment state
            state = env.render()  # (float array of shape [10])

            # Query the agent for its action decision
            action = self.agent.act(states=state)  # (scalar between 0 and 4)

            # Execute the decision and retrieve the current performance score
            state_1, reward, terminal, _ = env.step(action)  # (any scalar float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_only", help="Only train the AI single process...", default=False, action="store_true")
    parser.add_argument("--train", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--ppo", help="Train the AI", default=False, action="store_true")
    parser.add_argument("--random", help="Random Agent, default=False", action="store_true")
    parser.add_argument("--manhattan", help="Manhattan Agent", default=False, action="store_true")
    args = parser.parse_args()

    x = TrainWorker(1, "ppo")
    x.run()

from absl import flags, app

from deep_logistics.action_space import ActionSpace
from deep_logistics.agent import ManhattanAgent
from experiments.experiment_5.environment import Environment

from experiments.experiment_5.per_rl.agents.a2c import A2C
from experiments.experiment_5.per_rl.agents.ppo import PPO
from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE
FLAGS = flags.FLAGS

flags.DEFINE_boolean("callgraph", True, help="Creates a callgraph of the algorithm")

import gym
import gym_deep_logistics.gym_deep_logistics
import os

import pathos.multiprocessing as mp
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# https://github.com/ADGEfficiency/dsr-rl/blob/master/PITCHME.md
# MY NOTES: https://hastebin.com/usasisifuw
def main(argv):
    benchmark = False
    episodes = 13000000
    env_name = "CartPole-v0"
    #env_name = "deep-logistics-normal-v0"

    def submit(args):
        AGENT, spec, episodes = args
        agent = AGENT(**spec)

        # Number of epochs to do "supervised" learning (knowledge injection)
        curriculum_epochs = 5

        env = Environment(env_name)

        # Control set for manhattan distance
        manhattan_control = ManhattanAgent.automate
        is_deep_logisitcs =  "deep-logistics" in env_name

        while env.episode < episodes:

            action = agent.predict(env.state)

            if agent.epoch < curriculum_epochs and is_deep_logisitcs:
                action = manhattan_control(env.env.agent, perform_action=False)
                agent.data["action"] = tf.one_hot(action, agent.action_space)

            state1, reward, terminal = env.step(action)
            if reward >= 0.6 and is_deep_logisitcs:
                terminal = True

            agent.observe(
                obs1=state1,
                reward=reward,
                terminal=terminal
            )

    env = gym.make(env_name)

    if not benchmark:
        submit((PPO, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            tensorboard_enabled=True,
            baseline="reward_mean",
            batch_size=200,  # 2048
            mini_batches=4,  # 32
        ), episodes))
    else:
        agents = [
            [REINFORCE, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,
                batch_mode="episodic",
                tensorboard_enabled=True,
                tensorboard_path="./tb/"
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_mode="episodic",
                batch_size=64,
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Episodic"
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                batch_mode="steps",
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Steps64",
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                batch_mode="steps",
                policies=dict(
                    target=lambda agent: A2CPolicy(
                        agent=agent,
                        inference=True,
                        training=True,
                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
                    )
                ),
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="SinglePolicy",
            )],
            [PPO, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                tensorboard_enabled=True
            )]
        ]

        with mp.Pool(os.cpu_count()) as p:
            p.map(submit, [x + [episodes] for x in agents])



if __name__ == "__main__":

    app.run(main)

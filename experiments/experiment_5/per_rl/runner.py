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
    episodes = 130454535345
    env_name = "CartPole-v0"
    #env_name = "deep-logistics-normal-v0"

    def submit(args):
        AGENT, spec, episodes = args
        agent = AGENT(**spec)

        # Number of epochs to do "supervised" learning (knowledge injection)
        curriculum_epochs = 0

        env = Environment(env_name)
        agent.env = env

        # Control set for manhattan distance
        manhattan_control = ManhattanAgent.automate
        is_deep_logisitcs = "deep-logistics" in env_name

        while env.episode < episodes:

            action = agent.predict(env.state)

            if agent.epoch < curriculum_epochs and is_deep_logisitcs:
                action = manhattan_control(env.env.agent, perform_action=False)
            state1, reward, terminal, info = agent.step(action)

            if is_deep_logisitcs and terminal:
                agent.metrics.add("deliveries", info["deliveries"], ["sum"], "game", episode=True, epoch=False, total=False)
                agent.metrics.add("pickups", info["pickups"], ["sum"], "game", episode=True, epoch=False, total=False)

            agent.observe(
                actions=tf.one_hot(action, agent.action_space),
                rewards=reward,
                terminals=terminal
            )

            agent.train()

    if not benchmark:
        submit((PPO, dict(
            env=env_name,
            tensorboard_enabled=True,
            baseline="reward_mean",
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

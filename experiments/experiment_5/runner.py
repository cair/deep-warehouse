from absl import flags, app

from experiments.experiment_5.environment import Environment
from experiments.experiment_5.network import A2CPolicy

FLAGS = flags.FLAGS

flags.DEFINE_boolean("callgraph", True, help="Creates a callgraph of the algorithm")

import gym
import os
from experiments.experiment_5.a2c import A2C
from experiments.experiment_5.reinforce import REINFORCE
import pathos.multiprocessing as mp
import tensorflow as tf
import numpy as np
from experiments.experiment_5.ppo import PPO

tf.config.gpu.set_per_process_memory_growth(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# https://github.com/ADGEfficiency/dsr-rl/blob/master/PITCHME.md
def main(argv):
    benchmark = False
    episodes = 100000
    env_name = "CartPole-v0"

    test = [
        [0, 1, 2,3 ,4 ,5],
        [0, 1, 2,3 ,4 ,5]
    ]

    def submit(args):
        AGENT, spec, episodes = args
        agent = AGENT(**spec)

        env = Environment(env_name)

        while env.episode < episodes:
            action = agent.predict(env.state)
            state1, reward, terminal = env.step(action)
            agent.observe(
                obs1=state1,
                reward=reward,
                terminal=terminal
            )

        """try:
            #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            env = gym.make(env_name)



            for e in range(episodes):

                steps = 0
                terminal = False
                obs = env.reset()

                while not terminal:
                    action = agent.get_action(obs[None, :])
                    obs, reward, terminal, info = env.step(action)
                    reward = 0 if terminal else reward
                    agent.observe(obs, reward, terminal)
                    steps += 1
        except Exception as e:
            print(e)
            raise e"""

    env = gym.make(env_name)

    if not benchmark:
        """submit((REINFORCE, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            batch_mode="episodic",
            batch_size=64,
            tensorboard_enabled=True,
            tensorboard_path="./tb/"
        ), episodes))"""
        submit((REINFORCE, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            tensorboard_enabled=True,
            inference_only=False
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

import logging

import ray
import argparse
import os
import subprocess
import sys
from absl import flags, app
from tqdm import tqdm
#os.environ["path"] += ":/root"

flags.DEFINE_bool("dgx", False, "Run the experiment on DGX infrastructure.")
flags.DEFINE_bool("install", False, "Install dependencies automatically and exit")
flags.DEFINE_integer("train_epochs", 10, "Number of epochs to the train before demonstration.")
FLAGS = flags.FLAGS

from ray import tune
from ray.rllib.agents import ppo, impala
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from deep_logistics_ml.experiment_3.env import DeepLogisticsMultiEnv1


def install_dependencies():
    with open("requirements.txt", "r") as f:
        for package in f.readlines():
            subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])


def experiment_1():
    env = DeepLogisticsMultiEnv1(config=dict(
        graphics_render=False
    ))

    """Create distinct policy graphs for each agent."""
    policy_graphs = {
        k: (PPOPolicyGraph, env.observation_space, env.action_space, dict(
            gamma=0.95
        )) for k, a in env.agents.items()
    }

    policy_ids = list(policy_graphs.keys())

    trainer = impala.ImpalaAgent(env="DeepLogisticsMultiEnv1",
                            config=dict(
                                multiagent=dict(
                                    policy_graphs=policy_graphs,
                                    policy_mapping_fn=tune.function(
                                        lambda agent_id: agent_id
                                    )
                                ),
                                callbacks=dict(
                                    on_episode_end=tune.function(DeepLogisticsMultiEnv1.on_episode_end)
                                ),

                                num_envs_per_worker=288,
                                num_workers=288
                            ))

    while True:
        for _ in tqdm(range(FLAGS.train_epochs)):
            print(trainer.train())

        env.reset()
        terminal = False
        prev_action = {agent_id: None for agent_id in env.agents.keys()}
        prev_reward = {agent_id: None for agent_id in env.agents.keys()}
        prev_state = {agent_id: env.state_representation.generate(agent) for agent_id, agent in env.agents.items()}
        terminal_dict = {agent_id: False for agent_id in env.agents.keys()}
        while not terminal:

            for agent_id, agent in env.agents.items():
                if terminal_dict[agent_id]:
                    del prev_action[agent_id]
                    del prev_reward[agent_id]
                    del prev_state[agent_id]

                action = trainer.compute_action(prev_state[agent_id],
                                                prev_action=prev_action[agent_id],
                                                prev_reward=prev_reward[agent_id],
                                                policy_id=policy_ids[0]
                                                )
                prev_action[agent_id] = action

            prev_state, prev_reward, terminal_dict, info_dict = env.step(action_dict=prev_action)

            terminal = terminal_dict["__all__"]


def main(argv):
    """Argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgx", type=bool, default=False, help="Run the experiment on DGX infrastructure.")
    parser.add_argument("--install", type=bool, default=False, help="Install dependencies automatically and exit")
    args = parser.parse_args()

    #if args.dgx:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9,10,11,12"

    if args.install:
        install_dependencies()
        exit(0)

    ray.init(redis_address="cair-gpu04.uia.no:6379", logging_level=logging.DEBUG)

    experiment_1()



if __name__ == "__main__":
    # TODO cheat and symlink everything in root folder to dist packages.. xD
    for i in os.listdir("/root"):
        try:
            os.symlink("/root/%s" % i, "/usr/local/lib/python3.7/dist-packages/%s" % i)
        except:
            pass


    app.run(main)

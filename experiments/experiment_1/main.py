import sys
sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep-logistics")
sys.path.append("/home/per/GIT/code/deep-logistics")
sys.path.append("/root/deep-logistics")
from ray.rllib.agents.ppo import appo


from util import dict_merge
from ray import tune
import argparse
import alg_config
import os
import ray
from ray.rllib.agents import ppo, a3c, impala
from envs import DeepLogisticsA10M20x20D4



if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgx", help="Use DGX cpu and gpu", default=False, action="store_true")
    parser.add_argument("--ppo", help="Run PPO Experiment", default=False, action="store_true")
    parser.add_argument("--a3c", help="Use A3C Experiment", default=False, action="store_true")
    parser.add_argument("--appo", help="Use APPO Experiment", default=False, action="store_true")
    parser.add_argument("--test", help="Use Test Experiment", default=False, action="store_true")
    args = parser.parse_args()

    def on_episode_end(info):
        episode = info["episode"]
        env = info["env"].envs[0]

        deliveries = 0
        pickups = 0
        for agent in env.env.agents:
            deliveries += agent.total_deliveries
            pickups += agent.total_pickups

        deliveries = deliveries / len(env.env.agents)
        pickups = pickups / len(env.env.agents)

        episode.custom_metrics["deliveries"] = deliveries
        episode.custom_metrics["pickups"] = pickups


    experiments = {}
    custom_config = {
        "callbacks": {
            "on_episode_end": tune.function(on_episode_end)
        }
    }

    if args.dgx:
        custom_config["num_workers"] = os.cpu_count() - 4
        custom_config["num_gpus"] = 16
        custom_config["num_envs_per_worker"] = 1

    if args.ppo:
        config = ppo.DEFAULT_CONFIG.copy()
        dict_merge(config, custom_config)
        dict_merge(config, alg_config.ppo["v1"])
        experiments["ppo"] = {
            "run": "PPO",
            "env": DeepLogisticsA10M20x20D4,
            "stop": {
                "episode_reward_mean": 500
            },
            "config": config
        }

    if args.a3c:
        config = a3c.DEFAULT_CONFIG.copy()
        dict_merge(config, custom_config)

        if args.dgx:
            custom_config["num_workers"] = os.cpu_count() - 1
            custom_config["num_gpus"] = 4
            custom_config["num_envs_per_worker"] = 32

        #dict_merge(config, alg_config.ppo["v1"])
        experiments["a3c"] = {
            "run": "A3C",
            "env": DeepLogisticsA10M20x20D4,
            "stop": {
                "episode_reward_mean": 500
            },
            "config": config
        }

    if args.appo:
        config = appo.DEFAULT_CONFIG.copy()
        dict_merge(config, custom_config)

        if args.dgx:
            custom_config["num_workers"] = os.cpu_count() - 1
            custom_config["num_gpus"] = 16
            custom_config["num_envs_per_worker"] = 32

        #dict_merge(config, alg_config.ppo["v1"])
        experiments["appo"] = {
            "run": "APPO",
            "env": DeepLogisticsA10M20x20D4,
            "stop": {
                "episode_reward_mean": 500
            },
            "config": config
        }


    tune.run_experiments(experiments)

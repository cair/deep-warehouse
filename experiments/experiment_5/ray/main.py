import ray

from experiments.experiment_5.per_rl.agents.ppo import PPO
from experiments.experiment_5.ray.trainer import Trainer

if __name__ == "__main__":
    ray.init()

    trainer = Trainer(
        #env="deep-logistics-normal-v0",

        agent=PPO,
        agent_config=dict(
            env="CartPole-v0",
        ),
        num_workers=10
    )

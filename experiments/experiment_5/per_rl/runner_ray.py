import gym
import ray

from experiments.experiment_5.environment import Runner
from experiments.experiment_5.ppo import PPO

if __name__ == "__main__":
    ray.init(num_cpus=1)

    env_name = "CartPole-v0"
    env = gym.make(env_name)

    Runner().setup(
        env=env_name,
        agent=PPO,
        agent_config=dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            tensorboard_enabled=True
        ),
        num_actors=1
    )

#  """Debug flags."""
"""
        if self.debug_callgraph:
            self.debug_callgraph.start(reset=True)

        if self.debug_callgraph:
            self.debug_callgraph.done()
            print(self.debug_callgraph.output)
"""
#         self.debug_callgraph = pycallgraph.PyCallGraph(output=GraphvizOutput(output_file='filter_max_depth.png')) if FLAGS.callgraph else None
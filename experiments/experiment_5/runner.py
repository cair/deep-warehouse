import gym

import os
# Disable CUDA
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from experiments.experiment_5.pg import PGAgent

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    agent = PGAgent(
        obs_space=env.observation_space,
        action_space=env.action_space.n,
        batch_size=128,
        tensorboard_enabled=True,
        tensorboard_path="./tb/"
    )

    for e in range(1000):

        steps = 0
        terminal = False
        obs = env.reset()
        cum_loss = 0
        loss_n = 0

        while not terminal:
            action = agent.predict(obs[None, :])
            obs, reward, terminal, info = env.step(action)
            reward = 0 if terminal else reward
            agent.observe(reward, terminal)
            steps += 1

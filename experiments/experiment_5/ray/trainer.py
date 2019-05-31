import time
import numpy as np
import gym
import ray
import gym_deep_logistics.gym_deep_logistics
import logging
import tensorflow as tf

from experiments.experiment_5.environment import Environment


class RemoteAgent:

    def __init__(self, agent, agent_config):
        self.agent = agent(**agent_config)
        self.agent.env = Environment(self.agent.env)

    def set_weights(self, weights):
        self.agent.policy.master.set_weights(weights)


    def apply_gradients(self, gradients):
        self.agent.policy.master.optimizer.apply_gradients(
            zip(gradients, self.agent.policy.master.trainable_variables)
        )

        return 1

    def train_batch(self):

        state = self.agent.env.reset()
        while not self.agent.batch.ready():
            action = self.agent.predict(state)
            state1, reward, terminal, info = self.agent.step(action)
            self.agent.observe(
                actions=tf.one_hot(action, self.agent.action_space),
                rewards=reward,
                terminals=terminal
            )

        kwargs = {}
        kwargs["policy"] = self.agent.policy
        kwargs.update(self.agent._last_observation)

        # Retrieve batch
        batch = self.agent.batch.get()

        # Preprocess the data
        self.agent.preprocess(batch, ptype="batch", **kwargs)
        self.agent.batch.done()
        self.agent.udata.clear()
        self.agent.epoch += 1

        return batch

    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources)(cls)


class Trainer:

    def sync_weights(self):
        """
        Synchronize weights from local policy to all remote workers
        :return:
        """
        master_weights = self.policy.policy.master.get_weights()

        status = ray.get([worker.set_weights.remote(master_weights) for worker in self.workers])

    def collect_batch(self):
        """
        Collect experiences until the buffer is full.
        Then compute the gradients on all workers
        Then we retrieve all of the gradients and sum up and apply to the local policy copy
        :return:
        """
        batches = ray.get([w.train_batch.remote() for w in self.workers])
        return batches

        """if len(gradients) > 1:
            gradients = [tf.reduce_mean(grads, axis=0).numpy() for grads in zip(*gradients)]
        else:
            gradients = gradients[0]

        return gradients"""

    def train(self, batches):

        # Retrieve batch indices for this batch
        counter = len(batches[0]["inputs"])

        batch_indices = np.arange(counter)

        for epoch in range(self.policy.epochs):

            for batch in batches:
                #  Shuffle the batch indices if set
                if self.policy.batch_shuffle:
                    np.random.shuffle(batch_indices)

                # Construct batch data for the epoch
                batch_data = [{k: np.asarray(v)[batch_indices[i:i + self.policy.batch.batch_size]] for k, v in batch.items()} for i in range(0, counter, self.policy.batch.batch_size)]

                for b in batch_data:
                    losses, gradients = self.policy.compute_losses(**b)
                    self.policy.apply_gradients(gradients=gradients[0])
                    self.policy.policy.optimize()



    def apply_gradients(self, gradients):
        """self.policy.policy.master.optimizer.apply_gradients(
            zip(gradients, self.policy.policy.master.trainable_variables)
        )"""
        self.policy.policy.master.set_grads(gradients)
        self.policy.policy.optimize()

        #ray.get([w.apply_gradients.remote(gradients) for w in self.workers])

    def __init__(self, agent, agent_config, num_workers):
        agent_config.update(policy_update=dict(
            double=False,
            n_trainers=1,
            interval=1,
            strategy="copy",
            type="weights"
        ))
        self.policy = agent(**agent_config)
        self.env = self.policy.env
        self.policy.predict(self.env.reset())

        self.workers = [
            RemoteAgent.as_remote().remote(agent, agent_config) for _ in range(num_workers)
        ]
        self.collect_batch()  # TODO this should be initializer of weights

        while True:
            self.sync_weights()
            batches = self.collect_batch()
            self.train(batches=batches)


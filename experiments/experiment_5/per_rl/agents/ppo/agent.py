import numpy as np
import tensorflow as tf
from experiments.experiment_5.per_rl.agents.agent import Agent, DecoratedAgent
from experiments.experiment_5.per_rl.agents.ppo import defaults
from experiments.experiment_5.per_rl.agents.ppo.losses import PPOLosses
from experiments.experiment_5.per_rl.distribution.categorical import Categorical


@DecoratedAgent
class PPOAgent(Agent, PPOLosses):
    PARAMETERS = [
        "normalize_advantages",
        "gae_lambda",
        "gae",
        "epsilon",
        "kl_coeff",
        "vf_coeff",
        "vf_clipping",
        "vf_clip_param",
        "gamma",
        "entropy_coef"
    ]
    DEFAULTS = defaults.PPO

    def __init__(self, **kwargs):
        super(PPOAgent, self).__init__(**kwargs)

        self.add_processor("advantages", self.advantages, "batch")
        self.add_processor("neglogp", self.mb_neglogp, "mini-batch")


        #self.add_processor("update_kl", self.post_update_kl, "post")

        self.add_processor("policy_loss", self.policy_loss, "loss", text="Policy loss of PPO")
        self.add_processor("value_loss", self.value_loss, "loss", text="Action loss of PPO")
        self.add_processor("entropy_loss", self.entropy_loss, "loss", text="Action loss of PPO")
        self.add_callback(self.on_terminal, "on_terminal")

    def on_terminal(self):
        self.metrics.add("predicted_value", self.data["old_values"], ["mean"], "reward", episode=True, epoch=False,
                         total=False)

    def _predict(self, inputs):
        pred = super()._predict(inputs)
        logits = pred["logits"]

        sampled = tf.squeeze(tf.random.categorical(logits, 1))
        return sampled.numpy()

    def mb_neglogp(self, old_logits, logits, actions, **kwargs):

        return dict(
            neglogpac_old=Categorical.neglogpac(old_logits, actions),
            neglogpac_new=Categorical.neglogpac(logits, actions)
        )

    def generalized_advantage_estimation(self, old_values, last_obs, policy, rewards, terminals, **kwargs):
        V = np.concatenate((old_values, [policy([last_obs])["values"]]))
        terminal = np.concatenate((terminals, [0]))
        gamma = self.args["gamma"]
        lam = self.args["gae_lambda"]
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.batch.counter)):
            nextnonterminal = 1 - terminal[t + 1]
            delta = rewards[t] + gamma * (V[t + 1] * nextnonterminal) - V[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        returns = adv + old_values

        return dict(
            advantages=adv,
            returns=returns
        )

    def discounted_returns(self, rewards, terminals, old_values, **kwargs):
        discounted_rewards = np.zeros_like(rewards)
        advantage = np.zeros_like(rewards)
        cum_r = 0
        for i in reversed(range(0, self.batch.counter)):
            cum_r = rewards[i] + cum_r * self.args["gamma"] * (1 - terminals[i])

            discounted_rewards[i] = cum_r
            advantage[i] = cum_r + old_values[i]

        return dict(
            advantages=advantage,
            returns=discounted_rewards
        )

    def advantages(self, **kwargs):

        data = self.generalized_advantage_estimation(**kwargs) if self.args["gae"] else \
            self.discounted_returns(**kwargs)

        if self.args["normalize_advantages"]:
            data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (data["advantages"].std() + 1e-8)

        return data


    def post_update_kl(self, **kwargs):
        return {

        }

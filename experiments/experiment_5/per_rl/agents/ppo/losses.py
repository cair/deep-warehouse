import tensorflow as tf

from experiments.experiment_5.per_rl.distribution.categorical import Categorical


class PPOLosses:

    # @KeepLocals(
    #    include=["kl"],
    #    attribute="udata"
    # )
    def kl_loss(self, old_logits, logits, actions, **kwargs):
        # action_kl = tf.losses.kullback_leibler_divergence(neglogp_old, neglogp_new)
        # print(action_kl)
        # Metrics
        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac_new - neglogpac_old))
        # self.metrics.add("approxkl", approxkl, ["mean"], "train", epoch=True)

        logp = tf.math.softmax(logits, axis=1) * actions
        logq = tf.math.softmax(old_logits, axis=1) * actions

        kl = tf.reduce_mean(tf.reduce_sum(tf.math.softmax(logits) * (logp - logq), axis=1))

        # action_kl = tf.reduce_mean(tf.square(neglogp_new - neglogp_old))  # APPROX VERSION?

        # print(action_kl)
        return kl * self.args["kl_coeff"]

    def value_loss(self, old_values, values, returns, **kwargs):

        if self.args["vf_clipping"]:
            v_pred_clipped = \
                old_values + tf.clip_by_value(
                    values - old_values,
                    -self.args["vf_clip_param"],
                    self.args["vf_clip_param"]
                )
            vf_losses_1 = tf.square(values - returns)
            vf_losses_2 = tf.square(v_pred_clipped - returns)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses_1, vf_losses_2))

            # Metrics
            vf_clipfrac = tf.reduce_mean(
                tf.cast(tf.greater(tf.abs(v_pred_clipped - 1.0), self.args["vf_clip_param"]), dtype=tf.float64))
            self.metrics.add("vf_clipfrac", vf_clipfrac, ["mean"], "train", epoch=True)

        else:
            vf_loss = tf.losses.mean_squared_error(returns, values)

        return vf_loss * self.args["vf_coeff"]

    def policy_loss(self, neglogpac_old, neglogpac_new, advantages, **kwargs):

        surr1 = tf.exp(tf.stop_gradient(neglogpac_old) - neglogpac_new)
        surr2 = tf.clip_by_value(
            surr1,
            1.0 - self.args["epsilon"],
            1.0 + self.args["epsilon"]
        )

        surr_clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(surr1 - 1.0), self.args["epsilon"]), dtype=tf.float64)
        )

        self.metrics.add("policy_clipfrac", surr_clip_frac, ["mean"], "train", epoch=True)

        return -tf.reduce_mean(tf.minimum(
            surr1 * advantages,
            surr2 * advantages
        ))

    def entropy_loss(self, logits, **kwargs):
        return self.args["entropy_coef"] * -tf.reduce_mean(Categorical.entropy(logits))


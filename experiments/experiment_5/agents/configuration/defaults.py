
import tensorflow as tf

from experiments.experiment_5.agents.configuration.models import PGPolicy, A2CPolicy, PPOPolicy

REINFORCE = dict(
    batch_mode="episodic",
    batch_size=32,
    policies=dict(
        target=lambda agent: PGPolicy(
            agent=agent,
            inference=True,
            training=True,
            optimizer=tf.keras.optimizers.Adam(lr=0.001)
        )
    )
)

A2C = dict(
    batch_mode="steps",
    batch_size=32,
    mini_batches=8,
    entropy_coef=0.001,
    policies=dict(
        step=lambda agent: A2CPolicy(
            agent=agent,
            inference=True,
            training=False,
            optimizer=None
        ),
        train=lambda agent: A2CPolicy(
            agent=agent,
            inference=False,
            training=True,
            optimizer=tf.keras.optimizers.Adam(lr=0.001)  # decay=0.99, epsilon=1e-5)

        )
    ),
    policy_update=dict(
        interval=5,  # Update every 5 training epochs,
        strategy="copy",  # "copy, mean"
    )
)

PPO = dict(
    batch_mode="steps",
    batch_size=64,
    entropy_coef=0.01,  # Entropy should be 0.0 for continous action spaces.  # TODO
    value_coef=0.5,
    value_loss="huber",
    max_grad_norm=None,
    baseline="reward_mean",
    policies=dict(
        # The training policy (The new one)
        target=lambda agent: PPOPolicy(
            agent=agent,
            inference=False,
            training=True,
            optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
        ),
        # The old policy (The inference one)
        old=lambda agent: PPOPolicy(
            agent=agent,
            inference=True,
            training=False,
            optimizer=tf.keras.optimizers.Adam(lr=0.001)
        ),
    ),
    policy_update=dict(
        interval=5,  # Update every 10 training epochs,
        strategy="copy"
    )
)

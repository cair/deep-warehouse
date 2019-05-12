import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import PGPolicy, A2CPolicy, PPOPolicy

REINFORCE = dict(
    batch_mode="episodic",
    batch_size=32,
    batch_shuffle=False,
    policy=lambda agent: PGPolicy(
        agent=agent,
        inference=True,
        training=True,
        optimizer=tf.keras.optimizers.Adam(lr=0.001)
    )
)

A2C = dict(
    batch_mode="steps",
    batch_size=64,
    mini_batches=1,
    entropy_coef=0.001,
    batch_shuffle=False,
    policy=lambda agent: A2CPolicy(
        agent=agent,
        inference=False,
        training=True,
        optimizer=tf.keras.optimizers.Adam(lr=0.001)  # decay=0.99, epsilon=1e-5)
    ),
    policy_update=dict(
        interval=5,  # Update every 5 training epochs,
        strategy="copy",  # "copy, mean"
    )
)

PPO = dict(
    batch_mode="steps",
    epochs=10,
    batch_size=256,  # 2048
    batch_shuffle=True,  # Shuffle the batch (mini-batch or not)
    mini_batches=8,  # 32
    entropy_coef=0.01,  # Entropy should be 0.0 for continous action spaces.  # TODO
    value_coef=0.5,
    gamma=0.99,
    value_loss="huber",
    grad_clipping=None,
    baseline="reward_mean",
    policy=lambda agent: PPOPolicy(
        agent=agent,
        dual=True,
        optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        n_trainers=3
    ),
    policy_update=dict(
        interval=5,  # Update every 10 training epochs,
        strategy="copy"
    )
)

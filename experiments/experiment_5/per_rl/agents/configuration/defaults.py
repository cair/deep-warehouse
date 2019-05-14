import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import PGPolicy, A2CPolicy, PPOPolicy

REINFORCE = dict(
    batch_mode="episodic",
    batch_size=32,
    batch_shuffle=False,
    policy=lambda agent: PGPolicy(
        agent=agent,
        dual=False,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        update=dict(
            interval=1,
            strategy="copy"
        )
    )
)

A2C = dict(
    batch_mode="steps",
    batch_size=64,
    mini_batches=1,
    entropy_coef=0.001,
    grad_clipping=None,
    batch_shuffle=False,
    policy=lambda agent: A2CPolicy(
        agent=agent,
        dual=True,
        n_trainers=1,
        optimizer=tf.keras.optimizers.Adam(lr=0.001), # decay=0.99, epsilon=1e-5)
        update=dict(
            interval=5,  # Update every 5 training epochs,
            strategy="copy",  # "copy, mean"
        )
    ),

)

PPO = dict(
    batch_mode="steps",
    epochs=1,
    batch_size=64,  # 2048
    batch_shuffle=True,  # Shuffle the batch (mini-batch or not)
    mini_batches=64,  # 32
    entropy_coef=0.0002,  # Entropy should be 0.0 for continous action spaces.  # TODO
    value_coef=0.5,
    gamma=0.99,
    value_loss="mse",
    grad_clipping=None,
    baseline=None,
    policy=lambda agent: PPOPolicy(
        agent=agent,
        dual=True,
        optimizer=dict(
            policy=tf.keras.optimizers.Adam(lr=5e-3, decay=1e-6),
            value=tf.keras.optimizers.RMSprop(lr=5e-3, decay=1e-6),
        ),
        n_trainers=1,
        update=dict(
            interval=1,
            strategy="copy"  # Or mean
        )
    )
)

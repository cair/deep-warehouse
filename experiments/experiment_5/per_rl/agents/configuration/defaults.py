import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import PGPolicy, A2CPolicy, PPOPolicy

REINFORCE = dict(
    batch_mode="episodic",
    batch_size=32,
    gamma=0.99,
    entropy_coef=0.001,
    baseline=None,
    batch_shuffle=False,
    policy=lambda agent: PGPolicy(
        agent=agent,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
    ),
    policy_update=dict(
        double=True,
        n_trainers=1,
        interval=1,
        strategy="copy",  # copy, mean
        type="weights"  # weights, gradients
    )
)

A2C = dict(
    batch_mode="steps",
    value_coef=0.5,  # For action_value_loss, we multiply by this factor
    value_loss="mse",
    tau=0.95,
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
    gae_lambda=0.95,  # Lambda in General Advantange Estimation
    epsilon=0.2,  # Clipping factor
    kl_coef=0.2,  # TODO
    entropy_coef=0.0002,  # Entropy should be 0.0 for continous action spaces.  # TODO
    value_coef=0.5,
    gamma=0.99,

    batch_size=64,  # 2048
    mini_batches=64,  # 32
    epochs=1,  # TODO
    batch_shuffle=True,  # Shuffle the batch (mini-batch or not)

    value_loss="mse",  # TODO

    grad_clipping=None,

    #baseline=None,

    policy=lambda agent: PPOPolicy(
        agent=agent,
        dual=True,
        optimizer=dict(
            policy=tf.keras.optimizers.Adam(lr=5e-3, decay=1e-6),
            value=tf.keras.optimizers.RMSprop(lr=5e-3, decay=1e-6),
        )
    ),
    policy_update=dict(
        n_trainers=1,
        interval=1,
        strategy="copy",  # copy, mean
        type="weights"  # weights, gradients
    )
)

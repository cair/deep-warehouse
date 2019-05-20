import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import PGPolicy, A2CPolicy, PPOPolicy
# https://mpi4py.readthedocs.io/en/stable/
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
        optimizer=tf.keras.optimizers.Adam(lr=0.001), # decay=0.99, epsilon=1e-5)
    ),
    policy_update=dict(
        double=True,
        n_trainers=1,
        interval=5,
        strategy="copy",  # copy, mean
        type="weights"  # weights, gradients
    )

)

PPO = dict(

    # Generalized Advantage Function # TODO
    gae=True,  #
    gae_lambda=0.95,

    # Returns
    gamma=0.99,

    # Policy coefficients
    epsilon=0.2,  # Policy clipping factor
    kl_coef=0.2,  # TODO
    kl_target=0.01,  # TODO
    entropy_coef=0.2,  # Entropy should be 0.0 for continous action spaces.  # TODO

    # Value coefficients
    vf_loss="mse",  # TODO
    vf_clipping=True,   # TODO not working properly?
    vf_clip_param=0.2,#15.0,
    vf_coeff=1.0,


    # Sampling and Training
    batch_mode="steps",
    batch_shuffle=True,
    batch_size=2048,  # 2048
    mini_batches=4,
    epochs=10,

    # Optimization
    grad_clipping=5,

    # Policy definition
    policy=lambda agent: PPOPolicy(
        agent=agent,
        optimizer=dict(
            policy=tf.keras.optimizers.Adam(lr=0.0007),
            value=tf.keras.optimizers.Adam(lr=0.0007),
        )
        #optimizer=tf.keras.optimizers.Adam(lr=0.0007)
    ),

    # Policy update settings
    policy_update=dict(
        double=True,
        n_trainers=1,
        interval=1,
        strategy="copy",  # copy, mean  # TODO wierd
        type="weights"  # weights, gradients  # TODO wierd
    )
)

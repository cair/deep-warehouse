from experiments.experiment_5.per_rl.agents.ppo.policy import PPOPolicy
import tensorflow as tf
PPO = dict(

    # Generalized Advantage Function
    gae=True,
    gae_lambda=0.95,
    normalize_advantages=True,

    # Returns
    gamma=0.99,

    # Policy coefficients
    epsilon=0.2,  # Policy clipping factor
    kl_coeff=0.0,  # TODO
    kl_target=0.01,  # TODO
    entropy_coef=2.5e-3,  # Entropy should be 0.0 for continous action spaces.  # TODO

    # Value coefficients
    vf_loss="mse",  # TODO
    vf_clipping=False,   # TODO not working properly?
    vf_clip_param=5.0,
    vf_coeff=0.5,


    # Sampling and Training
    buffer_mode="steps",
    buffer_size=2048,  # 2048
    batch_shuffle=True,
    batch_size=64,
    epochs=5,

    # Optimization
    grad_clipping=0.2,  # TODO.

    # Policy definition
    policy=lambda agent: PPOPolicy(
        agent=agent,
        #optimizer=dict(
        #    policy=tf.keras.optimizers.Adam(lr=3e-4),
        #    value=tf.keras.optimizers.Adam(lr=3e-4),
        #)
        optimizer=tf.keras.optimizers.Adam(lr=3e-4)
    ),

    # Policy update settings
    policy_update=dict(
        double=True,
        n_trainers=1,
        interval=2,
        strategy="copy",  # copy, mean  # TODO wierd
        type="weights"  # weights, gradients  # TODO wierd
    )
)

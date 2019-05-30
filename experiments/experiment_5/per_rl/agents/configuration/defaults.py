import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import PGPolicy, A2CPolicy
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

ppo = dict(
    v1={
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # GAE(lambda) parameter
        "lambda": 1.0,
        # GAMMA
        "gamma": 0.99,  # TODO what is this

        # Initial coefficient for KL divergence
        "kl_coeff": 0.2,
        # Size of batches collected from each worker
        "sample_batch_size": 2000,
        # Number of timesteps collected for each SGD round
        "train_batch_size": 30000,
        # Total SGD batch size across all devices for SGD
        "sgd_minibatch_size": 128,
        # Number of SGD iterations in each outer loop
        "num_sgd_iter": 30,
        # Stepsize of SGD
        "lr": 3e-4, #5e-5,
        # Learning rate schedule
        "lr_schedule": None,
        # Share layers for value function
        "vf_share_layers": False,
        # Coefficient of the value function loss
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer
        "entropy_coeff": 0.01,
        # PPO clip parameter
        "clip_param": 0.2,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount
        "grad_clip": None,
        # Target value for KL divergence
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation
        "observation_filter": "NoFilter", # "MeanStdFilter",  # TODO NoFilter ?
        # Uses the sync samples optimizer instead of the multi-gpu one. This does
        # not support minibatches.
        "simple_optimizer": False,
        # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
        # sampling tasks for performance but can waste a large portion of samples.
        "straggler_mitigation": False,
        # Model architecture
        "model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [128, 128, 128],
        },
        # Workers
        'num_workers': 7,
        'num_envs_per_worker': 4,
        'num_gpus': 1

    }
)

"""
        
        'gamma': 0.99,
        'use_gae': True,
        'lambda': 0.97,  # GAE_Lambda
        'kl_coeff': 0.2,  # Likelihood ratio clipping / KL divergence coeff
        "kl_target": 0.01,
        'entropy_coeff': 0.01,  # Must figure out what this is TODO


        # Size of batches collected from each worker
        "sample_batch_size": 128,  # Number of experiences collected before sending to the Train worker
        #"train_batch_size": 128,  # Number of batches collected before training   # TODO
        "sgd_minibatch_size": 128,
        'batch_mode': 'truncate_episodes',  # alternatively. complete_episodes', Truncate sends training batches before episode ends. the other ones does not...


        # PPO optimizer (Adam)
        "lr": 3e-4,

        # Number of SGD iterations in each outer loop
        "num_sgd_iter": 25,
        # Total SGD batch size across all devices for SGD


        "model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [128, 128, 128],
        },
        'num_workers': 7,
        'num_envs_per_worker': 4,
        'num_gpus': 1

"""

"""
DEFAULT_CONFIG = {
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.


    # Number of timesteps collected for each SGD round
    # Share layers for value function
    "vf_share_layers": False,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount
    "grad_clip": None,
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This does
    # not support minibatches.
    "simple_optimizer": False,
 
})
"""

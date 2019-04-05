from tensorforce.agents import PPOAgent

class AgentFactory:

    @staticmethod
    def create(t, env):
        try:
            return getattr(AgentFactory, t)(env)
        except Exception as e:
            raise NotImplementedError("The agent type %s does not exist!" % t)


    @staticmethod
    def ppo(env):
        return PPOAgent(
            states=dict(type='float', shape=env.state_representation.get_shape()),
            actions=dict(type='int', num_actions=env.env.action_space.N_ACTIONS),
            # Automatically configured network
            #network=dict(type='auto', size=32, depth=2, internal_rnn=True),
            network=[
                dict(type='dense', size=128),
                dict(type='dense', size=128),
                dict(type='dense', size=128)
            ],
            # Update every 5 episodes, with a batch of 10 episodes
            update_mode=dict(unit='episodes', batch_size=10, frequency=5),
            # Memory sampling most recent experiences, with a capacity of 2500 timesteps
            # (2500 > [10 episodes] * [200 max timesteps per episode])
            memory=dict(type='latest', include_next_states=False, capacity=250000),
            discount=0.99, entropy_regularization=0.01,
            # MLP baseline
            baseline_mode='states', baseline=dict(type='mlp', sizes=[32, 32]),
            # Baseline optimizer
            baseline_optimizer=dict(
                type='multi_step', optimizer=dict(type='adam', learning_rate=1e-3), num_steps=5
            ),
            gae_lambda=0.97, likelihood_ratio_clipping=0.2,
            # PPO optimizer
            step_optimizer=dict(type='adam', learning_rate=3e-4), # was -4
            # PPO multi-step optimization: 25 updates, each calculated for 20% of the batch
            subsampling_fraction=0.2, optimization_steps=25
        )
"""
summarizer=dict(directory="./board",
                            labels=[
                                "bernoulli",
                                "beta",
                                "categorical",
                                "distributions",
                                "dropout",
                                "entropy",
                                "gaussian",
                                "graph",
                                "loss",
                                "losses",
                                "objective-loss",
                                "regularization-loss",
                                "relu",
                                "updates",
                                "variables",
                                "actions",
                                "states",
                                "rewards"
                            ]
                            ),
"""

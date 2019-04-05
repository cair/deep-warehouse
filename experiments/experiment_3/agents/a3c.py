import time

from ray.rllib.agents import Agent, with_common_config
from ray.rllib.evaluation.tf_policy_graph import LearningRateSchedule
from ray.rllib.optimizers import AsyncGradientsOptimizer
from ray.rllib.utils.annotations import override

DEFAULT_CONFIG = with_common_config(dict(

))


class A3CPolicyGraph:

class A3CAgent(Agent):

    _agent_name ="A3C-Per"
    _policy_graph = None # TODO


    @override(Agent)
    def _init(self, config, env_creator):

        policy_cls = self._policy_graph
        self.local_evaluator = self.make_local_evaluator(env_creator, policy_cls)
        self.remote_evaluators = self.make_remote_evaluators(env_creator, policy_cls, config["num_workers"])
        self.optimizer = self._make_optimizer()

    @override(Agent)
    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        start = time.time()
        while time.time() - start < self.config["min_iter_time_s"]:
            self.optimizer.step()

        result = self.collect_metrics()
        result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -prev_steps)


    def _make_optimizer(self):
        return AsyncGradientsOptimizer(self.local_evaluator,
                                       self.remote_evaluators,
                                       self.config["optimizer"])

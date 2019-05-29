import sys

from deep_logistics.scheduler import OnDemandScheduler
from deep_logistics.spawn_strategy import LocationSpawnStrategy
from experiments.experiment_3.state_representations import State0

sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep_logistics")
sys.path.append("/home/per/GIT/code/deep_logistics")
sys.path.append("/root")
from deep_logistics.environment import Environment
from deep_logistics.agent import InputAgent

if __name__ == "__main__":
    env =  Environment(
        height=5,
        width=5,
        depth=3,
        ups=None,
        ticks_per_second=1,
        taxi_n=1,
        taxi_agent=InputAgent,
        taxi_respawn=False,
        taxi_control="constant",
        scheduler=OnDemandScheduler,
        delivery_locations=None,
        spawn_strategy=LocationSpawnStrategy,
        graphics_render=True,
        graphics_tile_height=64,
        graphics_tile_width=64
    )

    env.deploy_agents()
    env.task_assignment()
    state = State0(env)
    agent = env.agents[0]

    def on_event():
        env.update()
        y = state.generate(agent)
        print(" - ".join([str(x) for x in y]))

    agent.add_event_callback(on_event)

    while True:
        agent.automate()
        env.render()

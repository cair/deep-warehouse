import sys
sys.path.append("/home/per/GIT/deep-logistics")
sys.path.append("/home/per/IdeaProjects/deep_logistics")
sys.path.append("/home/per/GIT/code/deep_logistics")
sys.path.append("/root")
from deep_logistics.environment import Environment
from deep_logistics.agent import InputAgent
from state_representations import State0

if __name__ == "__main__":
    env = Environment(
        height=10,
        width=10,
        depth=3,
        agents=2,
        agent_class=InputAgent,
        draw_screen=True,
        tile_height=32,
        tile_width=32,
        #scheduler=RandomScheduler,
        ups=60,
        ticks_per_second=1,
        spawn_interval=1,  # In steps
        task_generate_interval=1,  # In steps
        task_assign_interval=1,  # In steps
        delivery_points=[
            (7, 2),
            (2, 2),
            (2, 7),
            (7, 7)
        ],
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

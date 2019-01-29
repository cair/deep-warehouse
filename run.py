import asyncio
from multiprocessing import Process
from threading import Thread

import renderer
from agent import ManhattanAgent
from environment import Environment
from scheduler import RandomScheduler


if __name__ == "__main__":

    r = renderer.HTTPRenderer(loop=asyncio.new_event_loop(), fps=30)
    t = Thread(target=r._loop.run_forever)
    t.daemon = True
    t.start()

    env = Environment(
        height=50,
        width=50,
        depth=3,
        agents=5,
        agent_class=ManhattanAgent,
        renderer=r,
        tile_height=16,
        tile_width=16,
        scheduler=RandomScheduler
    )

    agent = env.add_agent(ManhattanAgent)

    while True:
        for agent in env.agents:

            agent.automate()
            env.update()

            r.blit(env.preprocess())

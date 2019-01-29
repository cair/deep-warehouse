import asyncio
import ctypes
from threading import Thread

import renderer
from action_space import ActionSpace
from agent import ManhattanAgent
from environment import Environment
from scheduler import RandomScheduler


if __name__ == "__main__":

    rt = renderer.HTTPRenderer(loop=asyncio.new_event_loop(), fps=30)

    env = Environment(
        height=25,
        width=25,
        depth=3,
        agents=1,
        agent_class=ManhattanAgent,
        renderer=rt,
        tile_height=16,
        tile_width=16,
        scheduler=RandomScheduler
    )

    rt.set_shared_state(*env.get_shared_state_pointer())
    rt = Thread(target=rt._loop.run_forever)
    rt.daemon = True
    rt.start()

    while True:
        for agent in env.agents:

            agent.automate()
            env.update()
            env.render()

            #r.blit(env.preprocess())

import asyncio
from threading import Thread

import renderer
from agent import ManhattanAgent
from environment import Environment
from scheduler import RandomScheduler




if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    r = renderer.HTTPRenderer(loop=asyncio.new_event_loop(), fps=30)
    t = Thread(target=r._loop.run_forever)
    t.daemon = True
    t.start()



    env = Environment(
        loop=loop,
        height=50,
        width=50,
        depth=3,
        agents=100,
        agent_class=ManhattanAgent,
        renderer=r,
        tile_height=16,
        tile_width=16,
        scheduler=RandomScheduler
    )

    agent = env.add_agent(ManhattanAgent)


    async def game_loop():
        # Play the game for X episodes
        while True:
            for agent in env.agents:
                await agent.automate()
                await env.update()

                await r.blit(await env.preprocess())
                #await asyncio.sleep(0.05)

    loop.create_task(game_loop())

    try:
        loop.run_forever()
    except KeyboardInterrupt as e:
        #print("Exited cleanly using keyboard.")
        exit(0)

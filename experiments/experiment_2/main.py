from deep_logistics import DeepLogistics
from deep_logistics import SpawnStrategies
from deep_logistics.agent import Agent, ManhattanAgent
if __name__ == "__main__":

    env = DeepLogistics(width=30,
                        height=30,
                        depth=3,
                        taxi_n=0,
                        ups=5000,
                        graphics_render=True,
                        delivery_locations=[
                            (5, 5),
                            (15, 15),
                            (20, 20),
                            (10, 10),
                            (5, 10)
                        ],
                        spawn_strategy=SpawnStrategies.RandomSpawnStrategy
                        )

    """Parameters"""
    EPISODES = 1000
    EPISODE_MAX_STEPS = 100

    """Add agents"""
    env.agents.add_agent(ManhattanAgent, n=20)

    for episode in range(EPISODES):
        env.reset()

        terminal = False
        steps = 0

        while terminal is False:
            env.update()
            env.render()

            terminal = env.is_terminal()
            steps += 1

            if terminal:
                print("Episode %s, Steps: %s" % (episode, steps))
                break

        """Add a new agent. (Harder) """
        #env.agents.add_agent(ManhattanAgent)







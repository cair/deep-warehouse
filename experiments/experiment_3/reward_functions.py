from deep_logistics.agent import Agent


def Reward0(player):

    if player.state in [Agent.IDLE, Agent.MOVING]:
        reward = 0.0
        terminal = False
    elif player.state in [Agent.PICKUP]:
        reward = .5 # Set back? TODO
        terminal = False
    elif player.state in [Agent.DELIVERY]:
        reward = 1
        terminal = False
    elif player.state in [Agent.DESTROYED]:
        reward = -1
        terminal = True
    elif player.state in [Agent.INACTIVE]:
        reward = 0
        terminal = True
    else:
        raise NotImplementedError("Should never happen. all states should be handled somehow")
    return reward, terminal

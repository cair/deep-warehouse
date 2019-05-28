from deep_logistics.agent import Agent


class RewardState:
    def __init__(self):
        self.last_x = None
        self.last_y = None

        self.is_closer = False
        self.counter = 0
    def update2(self):
        self.counter += 0.001
        return -self.counter
    def update(self, player):
        self.is_closer = False
        x = player.cell.x
        y = player.cell.y

        if player.task and self.last_x and self.last_y:
            task_coords = player.task.get_coordinates()
            d_x = x - task_coords.x
            d_y = y - task_coords.y
            d_x_old = self.last_x - task_coords.x
            d_y_old = self.last_y - task_coords.y

            dist = d_x + d_y
            dist_old = d_x_old + d_y_old

            if dist < dist_old:
                self.is_closer = True


        self.last_x = x
        self.last_y = y
        if self.is_closer:
            return 1
        else:
            return 0


rstate = RewardState()

def Reward0(player):

    if player.state in [Agent.IDLE, Agent.MOVING]:
        #reward = rstate.update2()
        reward = 0
        terminal = False
    elif player.state in [Agent.PICKUP]:
        rstate.counter = 0
        reward = 1  # Set back? TODO
        terminal = False
    elif player.state in [Agent.DELIVERY]:
        reward = 1
        terminal = False
    elif player.state in [Agent.DESTROYED]:
        reward = -10
        terminal = True
    elif player.state in [Agent.INACTIVE]:
        reward = 0
        terminal = True
    else:
        raise NotImplementedError("Should never happen. all states should be handled somehow")

    return reward, terminal

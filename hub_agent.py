
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import hub_carla
import time


actions = [[0., 0., 0.], [0., -0.1, 0], [0., -0.75, 0], [0., -0.5, 0], [0., -0.25, 0], [0., 0.25, 0], [0., 0.5, 0],
           [0., 0.75, 0], [0., 0.1, 0],
           [1., 0., 0], [1., -0.1, 0], [1., -0.75, 0], [1., -0.5, 0], [1., -0.25, 0], [1., 0.25, 0], [1., 0.5, 0],
           [1., 0.75, 0], [1., 0.1, 0],
           [0.75, 0., 0], [0.75, -0.1, 0.0], [0.75, -0.75, 0.0], [0.75, -0.5, 0.0], [0.75, -0.25, 0.0],
           [0.75, 0.25, 0.0],
           [0.75, 0.5, 0.0], [0.75, 0.75, 0.0], [0.75, 0.1, 0.0],
           [0.5, 0., 0], [0.5, -0.1, 0.0], [0.5, -0.75, 0.0], [0.5, -0.5, 0.0], [0.5, -0.25, 0.0], [0.5, 0.25, 0.0],
           [0.5, 0.5, 0.0], [0.5, 0.75, 0.0], [0.5, 0.1, 0.0],
           [0.25, 0., 0], [0.25, -0.1, 0], [0.25, -0.75, 0], [0.25, -0.5, 0], [0.25, -0.25, 0], [0.25, 0.25, 0],
           [0.25, 0.5, 0], [0.25, 0.75, 0], [0.25, 0.1, 0],
           [0., 0., 0.15], [0., 0., 0.25], [0., 0., 0.35], [0., 0., 0.5], [0., 0., 0.75], [0., 0., 1.0]]


class Net(nn.Module):
    def __init__(self, in_channels=3, n_actions=len(actions)):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc_backbone = nn.Sequential(
            nn.Linear(20, 4),
            nn.ReLU(),
            nn.Linear(4, 512 // 2),
            nn.ReLU(),
            nn.Linear(512 // 2, 512),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(272896, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x, m):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        m = m.float()
        x_v = self.fc_backbone(m)

        size = x.size(0)
        x_v = x_v.view(size, -1)
        x = x.view(size, -1)
        x = torch.cat([x, x_v], -1)

        x = F.relu(self.fc4(x))
        out3 = self.head(x)
        return out3


REPLAY_MEMORY_SIZE = 4000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 32
LR = 0.01
UPDATE_TARGER_EVERY = 10
EPISODES = 10000
epsilon = 0.9
epsilon_decy = 0.01
epsilon_min = 0.1
NUM_ACTION = len(actions)
MIN_REWARD = -100
FPS = 60.0

import pygame
# DQN agent class
class DQNAgent:
    gamma = 0.995

    def __init__(self, action_num=3):
        self.model = Net(n_actions=action_num)
        self.model.load_state_dict(torch.load('models/model.pkl'))

    def get_action(self, state, measurements, direction, episode):
        ran_num = epsilon - episode * epsilon_decy
        if ran_num < epsilon_min:
            ran_num = epsilon_min
        if random.random() <= ran_num:
            time.sleep(1 / FPS)
            return random.randint(0, NUM_ACTION - 1)
        else:
            state = torch.tensor(state).unsqueeze(0)
            measurements = torch.tensor(measurements).unsqueeze(0)
            value = self.model(state, measurements)
            # action=np.argmax(value)
            action_max_value, index = torch.max(value, 1)
            action = index.item()
            return action


def main():

    env = hub_carla.CarEnv()
    agent = DQNAgent(env.action_space_size)

    for episode in range(EPISODES):
        env.collision_hist = []
        clock = pygame.time.Clock()

        current_state, cur_measurements, direction = env.reset()

        while True:
            env.tick(clock)
            env.render(env.display)
            pygame.display.flip()
            action = agent.get_action(current_state, cur_measurements, direction, episode)
            new_state, _, done, measurements, direction = env.step(action)

            current_state = new_state
            cur_measurements = measurements

            if done:
                if env.arrive_target_location:
                    print("in episodes {}, agent arrive target location  ".format(episode))
                break
        for actor in env.actor_list:
            actor.destroy()
        env.destroy()


if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    main()
    pygame.quit()
    print('finish!')

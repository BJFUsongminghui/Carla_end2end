from collections import deque
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import end_to_end_carla
import time
import os


class Net(nn.Module):
    def __init__(self, in_channels=3, n_actions=3):
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
        self.fc4 = nn.Linear(532 * 512, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        size = x.size(0)
        out1 = x.view(size, -1)

        x = F.relu(self.fc4(out1))
        out3 = self.head(x)
        return out3


REPLAY_MEMORY_SIZE = 2000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 32
LR = 0.01
UPDATE_TARGER_EVERY = 100
EPISODES = 1000
epsilon = 0.9
epsilon_decy = 0.001
NUM_ACTION = 3
MIN_REWARD = -200


# DQN agent class
class DQNAgent:
    gamma = 0.995

    def __init__(self, action_num=3):
        self.model = Net(n_actions=action_num)
        if os.path.exists('models/model.pkl'):
            self.model.load_state_dict(torch.load('models/model.pkl'))
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = SummaryWriter('test_logs')

    def get_action(self, state, measurements, direction):

        state = torch.tensor(state).unsqueeze(0)
        measurements = torch.tensor(measurements).unsqueeze(0)
        value = self.model(state, measurements)
            # action=np.argmax(value)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if direction == 1:
            action = 4
        elif direction == 2:
            action = 24  # [0,1,0]
        elif direction == 3:
            action = 0
        return action


def play_in_rain():
    env = end_to_end_carla.CarEnv()
    weather = env.world.get_weather()
    weather.precipitation = 30  # 0 - 100
    env.world.set_weather(weather)
    agent = DQNAgent(action_num=env.action_space_size)
    arrive_num = 0.0
    for episode in range(EPISODES):
        env.collision_hist = []

        episode_reward = 0
        current_state, cur_measurements,direction= env.reset()
        current_state = env.reset()

        episode_start = time.time()
        while True:
            action = agent.get_action(current_state, cur_measurements, direction, episode)
            new_state, reward, done, measurements, direction = env.step(action)
            episode_reward += reward
            current_state = new_state

            if done:
                if env.arrive_target_location:
                    arrive_num += 1
                    print("in episodes {}, agent arrive target location  ".format(episode))
                break
        for actor in env.actor_list:
            actor.destroy()
        agent.tensorboard.add_scalar('reward', episode_reward, episode)
        if episode % 10 == 0:
            print("episodes {},  reward is {} spend time is {} ".format(episode, episode_reward,
                                                                        time.time() - episode_start))

    print("In rain day success rate is {}".format(arrive_num / EPISODES))



if __name__ == '__main__':
    play_in_rain()


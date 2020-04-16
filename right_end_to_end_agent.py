from collections import deque
from tensorboardX import SummaryWriter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import right_end_to_end_carla
import time
import os

actions = right_end_to_end_carla.actions


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
            nn.Linear(16, 4),
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


REPLAY_MEMORY_SIZE = 6000
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


# DQN agent class
class DQNAgent:
    gamma = 0.995

    def __init__(self, action_num=3):
        self.model = Net(n_actions=action_num)
        self.target_model = Net(n_actions=action_num)
        if os.path.exists('right_models/model.pkl'):
            self.model.load_state_dict(torch.load('right_models/model.pkl'))
        if os.path.exists('right_models/target_model.pkl'):
            self.target_model.load_state_dict(torch.load('right_models/target_model.pkl'))
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = SummaryWriter('right_end_to_end_logs')
        self.target_update_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.update_count = 0

    # save state action reward
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        # if replay memory size <MIN_REPLAY_MEMORY
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        reward = torch.tensor([t[2] for t in minibatch]).float()
        action = torch.tensor([t[1] for t in minibatch]).view(-1, 1).long()
        # get current state and trian get current_q
        current_state = torch.tensor([t[0] for t in minibatch]).float()
        cur_measurements = torch.tensor([t[5] for t in minibatch])
        cur_v = self.model(current_state, cur_measurements).gather(1, action)
        # get new_state and train get target_q
        new_state = torch.tensor([t[3] for t in minibatch]).float()
        done = torch.tensor([t[4] for t in minibatch]).int()

        measurements = torch.tensor([t[6] for t in minibatch])
        future_v = self.target_model(new_state, measurements).max(1)[0]
        target_v = reward + self.gamma * future_v * (1 - done)
        '''
        soft  q 

        next_value=torch.logsumexp(new_value,0)
        target_v=reward+(1-done)*self.gamma*next_value
        '''
        cur_v1 = cur_v.reshape(MINIBATCH_SIZE)
        loss = self.loss_func(cur_v1, target_v)

        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
        self.tensorboard.add_scalar('loss/value_loss', loss, self.update_count)
        self.update_count += 1

        # update target model every times
        self.target_update_counter += 1
        if self.target_update_counter % UPDATE_TARGER_EVERY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

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
    if not os.path.isdir('right_models'):
        os.makedirs('right_models')
    env = right_end_to_end_carla.CarEnv()
    agent = DQNAgent(env.action_space_size)

    for episode in range(EPISODES):
        env.collision_hist = []

        episode_reward = 0
        current_state, cur_measurements, direction = env.reset()

        episode_start = time.time()
        do_trian = 1
        while True:
            action = agent.get_action(current_state, cur_measurements, direction, episode)
            new_state, reward, done, measurements, direction = env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done, cur_measurements, measurements))
            if do_trian % 10 == 0:
                agent.train()
                time.sleep(0.01)
                do_trian = 0
            do_trian += 1

            current_state = new_state
            cur_measurements = measurements

            if done:
                if env.arrive_target_location:
                    print("in episodes {}, agent arrive target location  ".format(episode))
                break
        for actor in env.actor_list:
            actor.destroy()
        agent.tensorboard.add_scalar('reward', episode_reward, episode)
        if episode_reward >= MIN_REWARD:
            torch.save(agent.model.state_dict(), f'right_models/{episode_reward:>7.2f}reward.model')

        if episode % 10 == 0:
            print("episodes {},  reward is {} spend time is {} ".format(episode, episode_reward,
                                                                        time.time() - episode_start))
            torch.save(agent.model.state_dict(), 'right_models/model.pkl')
            torch.save(agent.target_model.state_dict(), 'right_models/target_model.pkl')

    torch.save(agent.model.state_dict(), 'right_models/model.pkl')


if __name__ == '__main__':
    main()

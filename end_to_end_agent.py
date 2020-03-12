from collections import deque
from tensorboardX import SummaryWriter
import random
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


REPLAY_MEMORY_SIZE = 3000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 32
LR = 0.01
UPDATE_TARGER_EVERY = 10
EPISODES = 10000
epsilon = 0.9
epsilon_decy = 0.001
epsilon_min=0.01
NUM_ACTION = 3
MIN_REWARD = -100


# DQN agent class
class DQNAgent:
    gamma = 0.995

    def __init__(self, action_num=3):
        self.model = Net(n_actions=action_num)
        self.target_model = Net(n_actions=action_num)
        if os.path.exists('models/model.pkl'):
            self.model.load_state_dict(torch.load('models/model.pkl'))
        if os.path.exists('models/target_model.pkl'):
            self.target_model.load_state_dict(torch.load('models/target_model.pkl'))
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = SummaryWriter('end_to_end_logs')
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
        cur_v = self.model(current_state).gather(1, action)
        # get new_state and train get target_q
        new_state = torch.tensor([t[3] for t in minibatch]).float()
        done=torch.tensor([t[4] for t in minibatch])
        future_v=self.target_model(new_state).max(1)[0]
        target_v=reward+self.gamma*future_v
        '''for index in range(len(reward)):
            if not done[index]:
                target_v[index]=reward[index]+self.gamma*future_v[index]
                print(target_v[index])

        with torch.no_grad():
            target_v1 = reward + self.gamma * self.target_model(new_state).max(1)[0]
        # update model
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

    def get_action(self, state, episode):
        ran_num=epsilon - episode * epsilon_decy
        if ran_num<epsilon_min:
            ran_num=epsilon_min
        if random.random() <= ran_num:
            return random.randint(0, NUM_ACTION - 1)
        else:
            state = torch.tensor(state).unsqueeze(0)
            value = self.model(state)
            # action=np.argmax(value)
            action_max_value, index = torch.max(value, 1)
            action = index.item()

            return action


def main():
    if not os.path.isdir('models'):
        os.makedirs('models')
    env = end_to_end_carla.CarEnv()
    agent = DQNAgent(action_num=env.action_space_size)

    for episode in range(EPISODES):
        env.collision_hist = []

        episode_reward = 0
        current_state = env.reset()

        episode_start = time.time()
        do_trian = 1
        while True:
            action = agent.get_action(current_state, episode)
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            if do_trian % 20 == 0:
                agent.train()
                do_trian = 0
            do_trian += 1

            current_state = new_state

            if done:
                if env.arrive_target_location:
                    print("in episodes {}, agent arrive target location  ".format(episode))
                break
        for actor in env.actor_list:
            actor.destroy()
        agent.tensorboard.add_scalar('reward', episode_reward, episode)
        if episode_reward >= MIN_REWARD:
            torch.save(agent.model.state_dict(), f'models/{episode_reward:>7.2f}reward.model')

        if episode % 10 == 0:
            print("episodes {},  reward is {} spend time is {} ".format(episode, episode_reward,
                                                                        time.time() - episode_start))
            torch.save(agent.model.state_dict(), 'models/model.pkl')
            torch.save(agent.target_model.state_dict(), 'models/target_model.pkl')

    torch.save(agent.model.state_dict(), 'models/model.pkl')


if __name__ == '__main__':
    main()

from collections import deque
from tensorboardX import SummaryWriter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import straight_vae_reward_carla as Carla_env
import time
import os
import vae.vae as vae

actions = Carla_env.actions


class Net(nn.Module):
    def __init__(self, in_channels=3, n_actions=len(actions)):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(Net, self).__init__()

        self.fc_backbone = nn.Sequential(
            nn.Linear(15, 4),
            nn.ReLU(),
            nn.Linear(4, 512 // 2),
            nn.ReLU(),
            nn.Linear(512 // 2, 512),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(1024, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x, m):
        m = m.float()
        x_v = self.fc_backbone(m)

        size = x.size(0)
        x_v = x_v.view(size, -1)
        x = x.view(size, -1)
        x = torch.cat([x, x_v], -1)

        x = F.relu(self.fc4(x))
        out3 = self.head(x)
        #   print(out3)
        return out3


REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 32
LR = 0.01
UPDATE_TARGER_EVERY = 10
EPISODES = 10000
epsilon = 0.9
epsilon_decy = 0.001
epsilon_min = 0.1
NUM_ACTION = len(actions)
MIN_REWARD = 0
FPS = 60.0


# DQN agent class
class DQNAgent:
    gamma = 0.8

    def __init__(self, action_num=3):
        self.model = Net(n_actions=action_num)

        self.target_model = Net(n_actions=action_num)

        if os.path.exists('straight_models/model.pkl'):
            self.model.load_state_dict(torch.load('straight_models/model.pkl'))
        if os.path.exists('straight_models/target_model.pkl'):
            self.target_model.load_state_dict(torch.load('straight_models/target_model.pkl'))
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = SummaryWriter('straight_end_to_end_logs')
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

        #future_v = self.target_model(new_state, measurements).max(1)[0]
        #target_v = reward + self.gamma * future_v * (1 - done)
        '''
        soft  q 
        '''
        new_value=self.target_model(new_state, measurements)
        next_value=torch.logsumexp(new_value,1)
        target_v=reward+(1-done)*self.gamma*next_value

        cur_v1 = cur_v.reshape(MINIBATCH_SIZE)
        # print('-----------------------------------------------')
        # print(cur_v1)
        # print(target_v)
        loss = self.loss_func(cur_v1, target_v)
        # print(loss)
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
            return random.randint(0, NUM_ACTION - 1)
        else:
            state = torch.tensor(state).unsqueeze(0)
            measurements = torch.tensor(measurements).unsqueeze(0)

            value = self.model(state, measurements)
            # action=np.argmax(value)
            action_max_value, index = torch.max(value, 1)
            print('-------------------max_value-----------')
            print(action_max_value)
            action = index.item()
            return action


def main():
    if not os.path.isdir('straight_models'):
        os.makedirs('straight_models')
    env = Carla_env.CarEnv()
    agent = DQNAgent(env.action_space_size)
    getvae = vae.GETVAE()

    for episode in range(EPISODES):
        env.collision_hist = []
        episode_reward = 0
        current_state, cur_measurements, direction = env.reset()
        current_vae_state = getvae.get_code(current_state)
        agent.train()
        episode_start = time.time()
        while True:
            action = agent.get_action(current_vae_state, cur_measurements, direction, episode)
            new_state, reward, done, measurements, direction = env.step(action)
            print(reward)
            episode_reward += reward

            current_vae_state = getvae.get_code(current_state)
            new_vae_state = getvae.get_code(new_state)
            agent.update_replay_memory(
                (current_vae_state, action, reward, new_vae_state, done, cur_measurements, measurements))
            current_vae_state = new_vae_state
            cur_measurements = measurements

            if done:
                if env.arrive_target_location:
                    print("in episodes {}, agent arrive target location  ".format(episode))
                break
        for actor in env.actor_list:
            actor.destroy()

        agent.tensorboard.add_scalar('reward', episode_reward, episode)
        if episode % 10 == 0:
            print("episodes {},  reward is {} spend time is {} ".format(episode, episode_reward,
                                                                        time.time() - episode_start))
            torch.save(agent.model.state_dict(), 'straight_models/model.pkl')
            torch.save(agent.target_model.state_dict(), 'straight_models/target_model.pkl')

    torch.save(agent.model.state_dict(), 'straight_models/model.pkl')


if __name__ == '__main__':
    main()

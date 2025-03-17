import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Env import SemiconductorEnv
import matplotlib.pyplot as plt


# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 将max_action转换为张量并移至与state相同的设备
        max_action_tensor = torch.tensor(self.max_action, dtype=torch.float32, device=state.device)
        action = torch.tanh(self.fc3(x)) * max_action_tensor
        return action

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

# DDPG 代理
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().data.numpy().flatten()
        return action

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        target_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, target_action)
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def main():
    env = SemiconductorEnv()
    state_dim = env.num_operation_types * 3  # 状态维度
    action_dim = 4  # 动作维度：setup_time, processing_time, left_operations, left_time
    max_action = np.array([70, 300, 20, 5000])  # 动作范围

    agent = DDPGAgent(state_dim, action_dim, max_action)

    num_episodes = 1000
    max_steps = 500

    # 训练成功标准参数

    success_window = 50  # 连续几个episode达标才算成功
    history_rewards = []  # 记录历史奖励
    convergence_threshold = 100.0  # 收敛阈值
    training_success = False  # 训练是否成功的标志


    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            noise = np.random.normal(0, 10, size=action_dim)
            action = np.clip(action + noise, 0, max_action)
            # print(f"Action: {action}")

            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            agent.train()

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        history_rewards.append(episode_reward)
        # 判断是否达到训练成功标准
        if episode >= success_window:
            recent_rewards = history_rewards[-success_window:]
            avg_recent_reward = np.mean(recent_rewards)
            
            # 标准2: 奖励收敛判断
            if episode >= 2*success_window:
                prev_rewards = history_rewards[-2*success_window:-success_window]
                avg_prev_reward = np.mean(prev_rewards)
                if abs(avg_recent_reward - avg_prev_reward) < convergence_threshold and avg_recent_reward > -50000:
                    training_success = True
                    print(f"训练收敛! 奖励稳定在{avg_recent_reward:.2f}")
                    break
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history_rewards)
    plt.title('Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('total reward')
    plt.grid(True)
    
    # 添加移动平均线以更清晰地显示趋势
    window_size = 10
    if len(history_rewards) >= window_size:
        moving_avg = np.convolve(history_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(history_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
        plt.legend()
    
    plt.savefig('reward_curve.png')  # 保存图片
    
    if training_success:
        print("训练成功！")
    else:
        print("达到最大训练轮数，但未满足收敛条件。")
    
if __name__ == "__main__":
    main()

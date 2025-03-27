import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envv import SemiconductorEnv, load_machines, load_jobs, load_operations
import matplotlib.pyplot as plt

##############################################################################
# OU 噪声类，用于连续动作空间的探索
class OUNoise:
    """
    Ornstein-Uhlenbeck噪声，用于连续动作的时间相关探索
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

##############################################################################
# 环境包装器
class RLEnvWrapper:
    def __init__(self):
        # 文件路径
        self.job_file = "dataset\\jobTypes.xlsx"
        self.machine_file = "dataset\\machineTypes.xlsx"
        self.operation_file = "dataset\\operationTypes.xlsx"
        self.problem_file = "dataset\\problem_one.xlsx"
        self.setup_file = "dataset\\setupTime.xlsx"

        # 初始化环境
        self.env = SemiconductorEnv(
            machines=load_machines(self.machine_file, self.problem_file),
            jobs=load_jobs(self.job_file, self.problem_file),
            operations=load_operations(self.operation_file, self.job_file)
        )
        
        # 设置状态和动作维度
        self.num_operation_types = len(set(op.operation_type_id for op in self.env.operation_instances))
        self.state_dim = self.num_operation_types * 3  # 三个状态向量
        self.action_dim = 4  # setup_time, processing_time, left_operations, left_time
    
    def reset(self):
        self.env.reset()
        return self.env.state()
    
    def execute_action(self, action):
        # 获取当前状态
        current_state = self.env.state()
        
        # 执行动作并获取setup_time和wait_time
        available_actions = self.env.get_available_actions()
        if not available_actions:
            # 如果没有可用动作，返回终止状态
            return current_state, 0, True, {}
        
        # 选择最接近的可用动作
        selected_action = self.env.select_action(action, available_actions)
        setup_time, wait_time = self.env.execute_action(selected_action)
        
        # 获取新状态
        next_state = self.env.state()
        
        # 计算奖励：负的时间消耗
        reward = -(setup_time + wait_time)
        
        # 检查是否完成
        done = self.env.is_done()
        
        return next_state, reward, done, {}

    def get_makespan(self):
        """
        获取当前所有机器的最大完成时间（makespan）
        """
        if hasattr(self.env, 'get_makespan'):
            return self.env.get_makespan()
        elif hasattr(self.env, 'machines'):
            # 如果环境中有machines属性，计算所有机器的最大完成时间
            return max([machine.completion_time for machine in self.env.machines]) if self.env.machines else 0
        else:
            # 如果无法直接获取，返回一个近似值
            return -self.env.state()[0] if hasattr(self.env, 'state') else 0

##############################################################################
# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)  # 增加网络宽度
        self.ln1 = nn.LayerNorm(400)  # 添加层归一化
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        max_action_tensor = torch.tensor(self.max_action, dtype=torch.float32, device=state.device)
        action = torch.tanh(self.fc3(x)) * max_action_tensor
        return action

##############################################################################
# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value

##############################################################################
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

##############################################################################
# DDPG 代理
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)  # 学习率可按情况微调

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 128
        self.gamma = 0.98
        self.tau = 0.005

        # 创建 OU 噪声用于动作探索
        self.ou_noise = OUNoise(action_dim)

    def select_action(self, state, explore=True):
        """
        默认不加噪声；如果需要在训练中进行探索，则设置 explore=True
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().data.numpy().flatten()

        if explore:
            # 加入 OU 噪声
            ou_sample = self.ou_noise.sample()
            action = np.clip(action + ou_sample, 0, self.max_action)
        else:
            action = np.clip(action, 0, self.max_action)

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

        # 计算目标Q值
        target_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, target_action)
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()

        # 计算当前Q值并更新 Critic
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 更新 Actor，使其最大化 Q
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_ou_noise(self):
        """在每个 episode 开始时重置 OU 噪声"""
        self.ou_noise.reset()

##############################################################################
# 主函数
def main():
    # 使用环境包装器
    env_wrapper = RLEnvWrapper()
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_dim
    max_action = np.array([1000, 1000, 1000, 1000])  # 动作范围

    agent = DDPGAgent(state_dim, action_dim, max_action)

    num_episodes = 2000
    max_steps = 10000

    # 训练成功标准参数
    success_window = 200
    history_rewards = []
    history_makespans = []  # 添加记录makespan的列表
    convergence_threshold = 100.0
    training_success = False

    # 训练过程中的 OU 噪声衰减参数
    # 初始噪声 sigma 设置为 0.2，不断衰减
    ou_sigma = 10
    ou_sigma_decay = 0.999
    min_ou_sigma = 2

    for episode in range(num_episodes):
        state = env_wrapper.reset()
        episode_reward = 0

        # 每轮开始时重置 OU 噪声
        agent.reset_ou_noise()

        # 随着训练进行，降低 OU 噪声幅度
        agent.ou_noise.sigma = max(ou_sigma, min_ou_sigma)

        for step in range(max_steps):
            # 选择带噪声动作
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env_wrapper.execute_action(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            agent.train()

            if done:
                break
        current_makespan = env_wrapper.get_makespan()

        # print(f"Episode {episode + 1}, Reward: {episode_reward}")
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Makespan: {current_makespan/60:.2f}h")
        history_rewards.append(episode_reward)
        history_makespans.append(current_makespan)
        
        # 衰减 OU 噪声的 sigma
        ou_sigma = ou_sigma * ou_sigma_decay

        # 判断收敛
        if episode >= success_window:
            recent_rewards = history_rewards[-success_window:]
            avg_recent_reward = np.mean(recent_rewards)
            
            if episode >= 2 * success_window:
                prev_rewards = history_rewards[-2 * success_window : -success_window]
                avg_prev_reward = np.mean(prev_rewards)
                if abs(avg_recent_reward - avg_prev_reward) < convergence_threshold and avg_recent_reward > -50000:
                    training_success = True
                    print(f"训练收敛! 奖励稳定在 {avg_recent_reward:.2f}")
                    break
                    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history_rewards)
    plt.title('RewardCurve')
    plt.xlabel('Episode')
    plt.ylabel('totalreward')
    plt.grid(True)

    # 添加移动平均线
    window_size = 10
    if len(history_rewards) >= window_size:
        moving_avg = np.convolve(history_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(history_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
        plt.legend()
    
    plt.savefig('rewardcurve.png')
    
    if training_success:
        print("训练成功！")
    else:
        print("达到最大训练轮数，但未满足收敛条件。")

if __name__ == "__main__":
    main()
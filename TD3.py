import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envv import SemiconductorEnv, load_machines, load_jobs, load_operations
import matplotlib.pyplot as plt
import os

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
        
        # 计算奖励：负的时间消耗和makespan影响
        base_reward = -(setup_time + wait_time)
        
        # 添加makespan相关奖励
        current_makespan = self.get_makespan()
        makespan_reward = -0.01 * current_makespan  # 轻微惩罚较高的makespan
        
        reward = base_reward + makespan_reward
        
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
            return max([machine.completion_time for machine in self.env.machines]) if self.env.machines else 0
        else:
            return -self.env.state()[0] if hasattr(self.env, 'state') else 0

##############################################################################
# Actor 网络 - 改为使用LayerNorm代替BatchNorm
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # 使用LayerNorm替代BatchNorm
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.fc4(x)) * torch.tensor(self.max_action, dtype=torch.float32, device=state.device)
        return action

##############################################################################
# Critic 网络 - 同样使用LayerNorm
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 架构
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # 使用LayerNorm替代BatchNorm
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 1)
        
        # Q2 架构 (双Q网络)
        self.fc5 = nn.Linear(state_dim + action_dim, 256)
        self.ln5 = nn.LayerNorm(256)  # 使用LayerNorm替代BatchNorm
        self.fc6 = nn.Linear(256, 256)
        self.ln6 = nn.LayerNorm(256)
        self.fc7 = nn.Linear(256, 128)
        self.ln7 = nn.LayerNorm(128)
        self.fc8 = nn.Linear(128, 1)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.ln1(self.fc1(x)))
        q1 = F.relu(self.ln2(self.fc2(q1)))
        q1 = F.relu(self.ln3(self.fc3(q1)))
        q1 = self.fc4(q1)
        
        # Q2
        q2 = F.relu(self.ln5(self.fc5(x)))
        q2 = F.relu(self.ln6(self.fc6(q2)))
        q2 = F.relu(self.ln7(self.fc7(q2)))
        q2 = self.fc8(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.ln1(self.fc1(x)))
        q1 = F.relu(self.ln2(self.fc2(q1)))
        q1 = F.relu(self.ln3(self.fc3(q1)))
        q1 = self.fc4(q1)
        
        return q1

##############################################################################
# 优先级经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.alpha = alpha  # 优先级的指数因子
        self.eps = 1e-6     # 小的常数防止优先级为0
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        
        return states, actions, rewards, next_states, dones, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < self.size:  # 确保索引有效
                self.priorities[idx] = priority + self.eps
    
    def size(self):
        return self.size

##############################################################################
# TD3 代理
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Critic网络 (双Q网络)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        # TD3超参数
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2  # 目标策略平滑噪声
        self.noise_clip = 0.5    # 噪声裁剪范围
        self.policy_freq = 2     # 策略更新频率(延迟更新)
        
        # 训练步数计数器
        self.total_it = 0
        
        # 噪声生成器
        self.ou_noise = OUNoise(action_dim)
        
    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).detach().cpu().numpy().flatten()
        
        if explore:
            # 使用OU噪声进行探索
            noise = self.ou_noise.sample()
            action = action + noise
            # 对不同维度使用适当的上限约束
            action = np.clip(action, 0, self.max_action)
        else:
            action = np.clip(action, 0, self.max_action)
            
        return action
            
    def train(self):
        # 经验回放缓冲区中的样本数量不足时，不进行训练
        if len(self.replay_buffer.buffer) < 5000:
            return
            
        self.total_it += 1
        
        # 从经验回放缓冲区中采样
        state, action, reward, next_state, done, indices = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            # 选择下一个动作并添加噪声进行平滑正则化
            noise = torch.FloatTensor(np.random.normal(0, self.policy_noise, size=(self.batch_size, self.action_dim))).to(self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            
            # 修复这里的clamp操作 - 使用逐元素的max/min代替clamp
            batch_size = next_action.shape[0]
            max_action_tensor = torch.tensor(self.max_action, dtype=torch.float32, device=self.device)
            max_action_expanded = max_action_tensor.unsqueeze(0).expand(batch_size, -1)
            
            # 使用max和min函数代替clamp
            next_action = torch.max(torch.min(next_action, max_action_expanded), torch.zeros_like(next_action))
            
            # 计算目标Q值 (取双Q网络中的较小值)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)
        
        # 计算Critic的损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 计算TD误差用于更新优先级
        with torch.no_grad():
            td_error = torch.abs(target_q - current_q1).cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_error)
        
        # 延迟更新Actor和目标网络
        if self.total_it % self.policy_freq == 0:
            # 更新Actor
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def reset_ou_noise(self):
        self.ou_noise.reset()
        
    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
        torch.save(self.actor_target.state_dict(), f"{path}_actor_target.pth")
        torch.save(self.critic_target.state_dict(), f"{path}_critic_target.pth")
        print(f"模型已保存到: {path}")
        
    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
        self.actor_target.load_state_dict(torch.load(f"{path}_actor_target.pth"))
        self.critic_target.load_state_dict(torch.load(f"{path}_critic_target.pth"))
        print(f"模型已加载自: {path}")

##############################################################################
# 主函数
def main():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # 使用环境包装器
    env_wrapper = RLEnvWrapper()
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_dim
    max_action = np.array([200, 300, 100, 1000])  # 动作范围
    
    agent = TD3Agent(state_dim, action_dim, max_action)
    
    num_episodes = 2000
    max_steps = 10000
    
    # 重启训练的次数
    max_restarts = 3
    restart_count = 0
    best_makespan = float('inf')
    
    success_window = 200
    history_rewards = []
    history_makespans = []
    all_history_rewards = []
    all_history_makespans = []
    convergence_threshold = 100.0
    
    # 记录每个重启阶段的收敛点
    convergence_points = []
    
    while restart_count <= max_restarts:
        print(f"=== 训练阶段 {restart_count+1}/{max_restarts+1} ===")
        
        # 如果是重新训练，重置噪声参数
        if restart_count > 0:
            # 重置奖励和makespan历史
            history_rewards = []
            history_makespans = []
            # 重置参数
            agent.policy_noise = 0.4  # 增加噪声促进探索
            ou_sigma = 2.0
        else:
            agent.policy_noise = 0.2
            ou_sigma = 2.0
            
        ou_sigma_decay = 0.997
        min_ou_sigma = 0.2
        
        for episode in range(num_episodes):
            state = env_wrapper.reset()
            episode_reward = 0
            
            agent.reset_ou_noise()
            agent.ou_noise.sigma = max(ou_sigma, min_ou_sigma)
            
            for step in range(max_steps):
                action = agent.select_action(state, explore=True)
                next_state, reward, done, _ = env_wrapper.execute_action(action)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                agent.train()
                
                if done:
                    break
            
            current_makespan = env_wrapper.get_makespan()
            
            print(f"阶段 {restart_count+1}, Episode {episode + 1}, Reward: {episode_reward:.2f}, Makespan: {current_makespan/60:.2f}h")
            history_rewards.append(episode_reward)
            history_makespans.append(current_makespan)
            all_history_rewards.append(episode_reward)
            all_history_makespans.append(current_makespan)
            
            ou_sigma = ou_sigma * ou_sigma_decay
            
            # 检查是否收敛
            if episode >= success_window:
                recent_rewards = history_rewards[-success_window:]
                avg_recent_reward = np.mean(recent_rewards)
                
                if episode >= 2 * success_window:
                    prev_rewards = history_rewards[-2 * success_window : -success_window]
                    avg_prev_reward = np.mean(prev_rewards)
                    
                    # 判断奖励是否稳定且有合理的值
                    if abs(avg_recent_reward - avg_prev_reward) < convergence_threshold and avg_recent_reward > -50000:
                        print(f"阶段 {restart_count+1} 训练收敛! 奖励稳定在 {avg_recent_reward:.2f}")
                        
                        # 记录收敛点
                        convergence_points.append((len(all_history_rewards), avg_recent_reward))
                        
                        # 如果当前makespan比历史最好的更好，保存为最佳模型
                        recent_makespan = np.mean(history_makespans[-success_window:])
                        if recent_makespan < best_makespan:
                            best_makespan = recent_makespan
                            agent.save_models("models/td3_best")
                            print(f"发现更好的模型! Makespan: {best_makespan/60:.2f}h")
                        
                        # 保存当前阶段模型
                        agent.save_models(f"models/td3_stage_{restart_count+1}")
                        break
            
            # 如果达到最大episode数但没有收敛
            if episode == num_episodes - 1:
                print(f"阶段 {restart_count+1} 达到最大训练轮数，但未满足收敛条件。")
                agent.save_models(f"models/td3_stage_{restart_count+1}_incomplete")
        
        restart_count += 1
        # 如果已经达到最大重启次数，退出循环
        if restart_count > max_restarts:
            break
            
    # 绘制总体奖励曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(all_history_rewards)
    plt.title('总体奖励曲线')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # 标记每次收敛点
    for i, (x, y) in enumerate(convergence_points):
        plt.axvline(x=x, color='r', linestyle='--', alpha=0.5)
        plt.text(x, min(all_history_rewards) * 0.9, f"阶段{i+1}收敛", rotation=90)
    
    # 绘制总体makespan曲线
    plt.subplot(2, 1, 2)
    plt.plot(all_history_makespans)
    plt.title('总体Makespan曲线')
    plt.xlabel('Episode')
    plt.ylabel('Makespan (分钟)')
    plt.grid(True)
    
    # 标记每次收敛点
    for i, (x, _) in enumerate(convergence_points):
        plt.axvline(x=x, color='r', linestyle='--', alpha=0.5)
        plt.text(x, max(all_history_makespans) * 0.9, f"阶段{i+1}收敛", rotation=90)
    
    plt.tight_layout()
    plt.savefig('td3_training_history.png')
    
    print(f"训练完成! 最佳Makespan: {best_makespan/60:.2f}h")
    print(f"最佳模型已保存到: models/td3_best")

if __name__ == "__main__":
    main()
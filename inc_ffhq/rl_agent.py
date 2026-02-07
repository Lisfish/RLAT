import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from my_models import DQN

# 定义经验回放的结构
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQNAgent:
    def __init__(self, input_size, output_size, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 维度参数
        self.input_size = input_size  # 状态维度 (例如 VGG 特征展平后的 25088)
        self.output_size = output_size  # 动作空间大小 (例如 100 个网格)

        # 2. 超参数 (从你的 config 中获取)
        self.epsilon = config.get('EPSILON', 1.0)
        self.eps_decay = config.get('EPS_DECAY', 0.995)
        self.eps_min = config.get('EPS_MIN', 0.1)
        self.gamma = config.get('GAMMA', 0.99)
        self.batch_size = config.get('BATCH_SIZE', 32)

        # 3. 网络初始化
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5)
        self.memory = deque(maxlen=2000)
        self.criterion = nn.MSELoss()

    def select_action(self, state, mode="train"):
        """根据 epsilon-greedy 策略选择动作"""
        if mode == "train" and random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)

        with torch.no_grad():
            # state 需要是 tensor 并移动到 device
            state = state.to(self.device).unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, next_state, reward):
        """存储经验到缓冲区"""
        self.memory.append(Transition(state, action, next_state, reward))

    def optimize_model(self):
        """从经验回放中采样并更新 Policy Network"""
        if len(self.memory) < self.batch_size:
            return None

        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)

        # 计算当前 Q 值 Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算目标 Q 值: V(s_{t+1}) = max Q_target(s_{t+1}, a)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 计算 Loss 并反向传播
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        return loss.item()

    def update_target_network(self):
        """同步 Target Network 权重"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
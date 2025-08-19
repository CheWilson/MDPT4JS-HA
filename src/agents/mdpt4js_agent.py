import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn.utils
from torch.utils.data import DataLoader, TensorDataset
from src.networks.multi_discrete_network import MultiDiscreteACTNetwork
from src.data.replay_buffer import ReplayBuffer


class MultiDiscreteACT4JSAgent:
    """MDPT4JS代理類 - 共享Transformer，三個動作頭，PPO(GAE, mini-batch)"""

    def __init__(self, input_dims, n_actions_list, *, lr=1e-3, gamma=0.9,
                 clip_epsilon=0.2, buffer_capacity=10000, buffer_size=256, mini_batch_size=32,
                 n_epochs=10, gae_lambda=0.95, entropy_coef=1e-2, total_acs=1000):
        """
        初始化MDPT4JS代理
        
        Args:
            input_dims: 輸入維度
            n_actions_list: 各層動作數量列表
            lr: 學習率
            gamma: 折扣因子
            clip_epsilon: PPO裁剪參數
            buffer_capacity: 緩衝區容量
            buffer_size: 緩衝區大小閾值
            mini_batch_size: 小批次大小
            n_epochs: 每次更新的epoch數
            gae_lambda: GAE參數
            entropy_coef: 熵正則化係數
            total_acs: 總AC數量
        """
        self.gamma = gamma
        self.clip_eps = clip_epsilon
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.total_acs = total_acs
        
        self.network = MultiDiscreteACTNetwork(input_dims, n_actions_list, lr, self.total_acs)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.device = self.network.device
        
        # 初始化損失追蹤列表
        self.all_losses = []

    def store_transition(self, state, actions, reward, next_state, done):
        """
        儲存transition到buffer
        
        Args:
            state: 當前狀態
            actions: 動作列表
            reward: 獎勵
            next_state: 下一個狀態
            done: 是否結束
        """
        self.memory.add(state, actions, reward, next_state, done)

    def should_learn(self):
        """檢查是否應該開始學習 - 當buffer達到容量時觸發"""
        return len(self.memory) >= self.buffer_size

    def compute_gae(self, rewards, values, next_values, dones):
        """
        計算GAE
        
        Args:
            rewards: 獎勵張量
            values: 價值張量
            next_values: 下一個價值張量
            dones: 結束標誌張量
            
        Returns:
            tuple: (advantages, returns)
        """
        advantages = T.zeros_like(rewards).to(self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def learn(self):
        """
        學習更新網絡參數
        
        Returns:
            float: 平均損失
        """
        if not self.should_learn():
            return 0.0

        # Dump buffer - 取出所有數據並清空
        states, actions_batch, rewards, next_states, dones = self.memory.dump_all()
        states = T.tensor(np.stack(states), dtype=T.float32).to(self.device)
        next_states = T.tensor(np.stack(next_states), dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.device)

        # Multi-discrete actions
        dc_act = T.tensor([a[0] for a in actions_batch], dtype=T.int64).to(self.device)
        server_act = T.tensor([a[1] for a in actions_batch], dtype=T.int64).to(self.device)
        ac_act = T.tensor([a[2] for a in actions_batch], dtype=T.int64).to(self.device)

        with T.no_grad():
            dists, values = self.network(states)
            old_dc_log = dists[0].log_prob(dc_act)
            old_server_log = dists[1].log_prob(server_act)
            old_ac_log = dists[2].log_prob(ac_act)
            values = values.squeeze()
            next_values = T.roll(values, shifts=-1)
            next_values[-1] = 0  # 最後一個next_value設為0
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # 創建dataset用於標準PPO mini-batch訓練
        dataset = TensorDataset(states, dc_act, server_act, ac_act,
                                advantages, returns,
                                old_dc_log, old_server_log, old_ac_log)
        
        # 使用較小的mini_batch_size進行標準PPO訓練
        loader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        total_loss = 0.0
        batch_count = 0
        
        # 標準PPO: 多個epochs，每個epoch遍歷所有mini-batches
        for _ in range(self.n_epochs):
            for (s, f_a, s_a, v_a, adv, ret, o_f, o_s, o_v) in loader:
                batch_count += 1
                
                dists, val = self.network(s)
                log_f = dists[0].log_prob(f_a)
                log_s = dists[1].log_prob(s_a)
                log_v = dists[2].log_prob(v_a)

                ratio = (T.exp(log_f - o_f) + T.exp(log_s - o_s) + T.exp(log_v - o_v)) / 3.0
                surr1 = ratio * adv
                surr2 = T.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -T.min(surr1, surr2).mean()
                entropy = (dists[0].entropy() + dists[1].entropy() + dists[2].entropy()).mean()
                actor_loss -= self.entropy_coef * entropy

                critic_loss = F.mse_loss(val.squeeze(), ret)
                loss = actor_loss + 0.5 * critic_loss

                self.network.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.network.optimizer.step()
                
                total_loss += loss.item()

        # 計算並記錄平均損失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        self.all_losses.append(avg_loss)
        
        return avg_loss
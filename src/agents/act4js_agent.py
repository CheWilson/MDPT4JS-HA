import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn.utils
from torch.utils.data import DataLoader, TensorDataset
from src.networks.transformer_network import ACTNetwork
from src.data.replay_buffer import ReplayBuffer
from src.training.gae_utils import compute_gae


class ACT4JSAgent:
    """ACT4JS代理類 - 使用三個獨立網絡，每個網絡用PPO(GAE, mini-batch)更新"""

    def __init__(self, input_dims, n_actions_list, *, lr=1e-3, gamma=0.9,
                 clip_epsilon=0.2, buffer_capacity=10000, buffer_size=256, mini_batch_size=32,
                 n_epochs=10, gae_lambda=0.95, entropy_coef=1e-2):
        """
        初始化ACT4JS代理
        
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
        """
        self.gamma = gamma
        self.clip_eps = clip_epsilon
        self.buffer_size = buffer_size
        self.mini_batch = mini_batch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        # 創建三個獨立網絡
        self.dc_net = ACTNetwork(input_dims, n_actions_list[0], lr)
        self.server_net = ACTNetwork(input_dims, n_actions_list[1], lr)
        self.ac_net = ACTNetwork(input_dims, n_actions_list[2], lr)

        # 為每個網絡創建獨立的經驗緩衝區
        self.dc_mem = ReplayBuffer(buffer_capacity)
        self.server_mem = ReplayBuffer(buffer_capacity)
        self.ac_mem = ReplayBuffer(buffer_capacity)
        
        # 損失記錄
        self.dc_losses = []
        self.server_losses = []
        self.ac_losses = []
        
        # 設置設備屬性，用於 choose_action_single
        self.device = self.dc_net.device

    def store_transition(self, state, actions, rewards, next_state, done):
        """
        儲存transition到各自的緩衝區
        
        Args:
            state: 當前狀態
            actions: 動作列表 [dc_action, server_action, ac_action]
            rewards: 獎勵列表 [dc_reward, server_reward, ac_reward]
            next_state: 下一個狀態
            done: 是否結束
        """
        self.dc_mem.add(state, actions[0], rewards[0], next_state, done)
        self.server_mem.add(state, actions[1], rewards[1], next_state, done)
        self.ac_mem.add(state, actions[2], rewards[2], next_state, done)

    def _compute_gae(self, rewards, values, next_values, dones, device):
        """
        計算Generalized Advantage Estimation
        
        Args:
            rewards: 獎勵張量
            values: 價值張量
            next_values: 下一個價值張量
            dones: 結束標誌張量
            device: 設備
            
        Returns:
            tuple: (advantages, returns)
        """
        advantages = T.zeros_like(rewards).to(device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _learn_network(self, net, memory):
        """
        更新指定網路的參數
    
        Args:
            net: 要更新的網路 (dc_net, server_net 或 ac_net)
            memory: 對應的經驗回放緩衝區
    
        Returns:
            float: 平均損失值
        """
        if len(memory) < self.buffer_size:
            return 0.0

        states, actions, rewards, next_states, dones = memory.dump_all()
        device = net.device
        states = T.tensor(np.stack(states), dtype=T.float32).to(device)
        next_states = T.tensor(np.stack(next_states), dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.int64).to(device)

        with T.no_grad():
            dist, values = net(states)
            old_log_probs = dist.log_prob(actions)
            values = values.squeeze()
            next_values = T.roll(values, shifts=-1)
        
        advantages, returns = self._compute_gae(rewards, values, next_values, dones, device)

        dataset = TensorDataset(states, actions, advantages, returns, old_log_probs)
        loader = DataLoader(dataset, batch_size=self.mini_batch, shuffle=True)

        total_loss = 0.0
        batch_count = 0
    
        for _ in range(self.n_epochs):
            for s, a, adv, ret, old_log in loader:
                dist, val = net(s)
                log_prob = dist.log_prob(a)
                ratio = T.exp(log_prob - old_log)
                surr1 = ratio * adv
                surr2 = T.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -T.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                actor_loss -= self.entropy_coef * entropy
                critic_loss = F.mse_loss(val.squeeze(), ret)
                loss = actor_loss + 0.5 * critic_loss

                net.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                net.optimizer.step()

                total_loss += loss.item()
                batch_count += 1
    
        # 計算平均損失並記錄
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    
        # 根據網路類型記錄損失
        if net == self.dc_net:
            self.dc_losses.append(avg_loss)
        elif net == self.server_net:
            self.server_losses.append(avg_loss)
        elif net == self.ac_net:
            self.ac_losses.append(avg_loss)
    
        return avg_loss

    def learn(self):
        """
        學習更新所有三個網絡
        
        Returns:
            float: 平均損失
        """
        f_loss = self._learn_network(self.dc_net, self.dc_mem)
        s_loss = self._learn_network(self.server_net, self.server_mem)
        v_loss = self._learn_network(self.ac_net, self.ac_mem)

        return (f_loss + s_loss + v_loss) / 3.0
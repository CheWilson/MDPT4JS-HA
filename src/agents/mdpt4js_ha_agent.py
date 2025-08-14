import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn.utils
from torch.utils.data import DataLoader, TensorDataset
from src.networks.multi_discrete_network import MultiDiscreteACT4JSPredNetwork
from src.data.replay_buffer import ReplayBuffer


class MultiDiscreteACT4JSPredAgent:
    """MDPT4JS-HA代理類 - 同步更新版本"""
    
    def __init__(self, input_dims, n_actions_list, *, lr=1e-3, gamma=0.9,
                 clip_epsilon=0.2, buffer_capacity=10000, buffer_size=256, mini_batch_size=32,
                 n_epochs=10, gae_lambda=0.95, entropy_coef=1e-2, total_acs=1000):
        """
        初始化MDPT4JS-HA代理
        
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
        
        self.network = MultiDiscreteACT4JSPredNetwork(input_dims, n_actions_list, lr, self.total_acs)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.device = self.network.device
        
        # 損失追蹤
        self.all_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.prediction_losses = []
        self.server_pred_dc_losses = []
        self.ac_pred_dc_losses = []
        self.ac_pred_server_losses = []

    def store_transition(self, state, actions, reward, next_state, done):
        """儲存transition到buffer"""
        self.memory.add(state, actions, reward, next_state, done)

    def should_learn(self):
        """檢查是否應該開始學習"""
        return len(self.memory) >= self.buffer_size

    def learn(self):
        """
        同步版本：PPO和預測頭使用相同數據同時更新
        完全按照改進的算法實現
        """
        # 算法檢查：if |D| >= β then
        if not self.should_learn():
            return 0.0
        
        # 取出所有數據（算法中的batch）
        states, actions_batch, rewards, next_states, dones = self.memory.dump_all()
        
        # 準備數據
        states = T.tensor(np.stack(states), dtype=T.float32).to(self.device)
        next_states = T.tensor(np.stack(next_states), dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.device)

        dc_actions = T.tensor([a[0] for a in actions_batch], dtype=T.int64).to(self.device)
        server_actions = T.tensor([a[1] for a in actions_batch], dtype=T.int64).to(self.device)
        ac_actions = T.tensor([a[2] for a in actions_batch], dtype=T.int64).to(self.device)

        # 1. PPO更新部分
        ppo_loss = self._synchronized_ppo_update(states, dc_actions, server_actions, 
                                                ac_actions, rewards, next_states, dones)
        
        # 2. 立即用相同數據更新預測頭
        pred_loss = self._synchronized_prediction_update(states, dc_actions, server_actions, ac_actions)
        
        # 3. 記錄總損失
        total_loss = ppo_loss + pred_loss
        self.all_losses.append(total_loss)
        
        return total_loss

    def _synchronized_ppo_update(self, states, dc_actions, server_actions, ac_actions, 
                            rewards, next_states, dones):
        """算法中的PPO更新部分 - 修改為標準PPO做法"""
    
        # 計算advantages和returns
        with T.no_grad():
            dists, _, values = self.network(states)
            values = values.squeeze()
            next_values = T.roll(values, shifts=-1)
            next_values[-1] = 0  # 最後一個next_value設為0
            old_dc_log = dists[0].log_prob(dc_actions)
            old_server_log = dists[1].log_prob(server_actions)
            old_ac_log = dists[2].log_prob(ac_actions)
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # 創建dataset用於mini-batch訓練
        dataset = TensorDataset(states, dc_actions, server_actions, ac_actions,
                          advantages, returns, old_dc_log, old_server_log, old_ac_log)
    
        loader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        total_ppo_loss = 0.0
        batch_count = 0

        # 算法：for K epochs do
        for _ in range(self.n_epochs):  # 10個 epochs
            for s, f_a, s_a, v_a, adv, ret, o_f, o_s, o_v in loader:  # mini-batches
                batch_count += 1
            
                # PPO更新邏輯保持不變
                self.network.optimizer.zero_grad()
            
                dists, _, val = self.network(s)
                dc_dist, server_dist, ac_dist = dists
            
                # 計算PPO損失
                log_f = dc_dist.log_prob(f_a)
                log_s = server_dist.log_prob(s_a)
                log_v = ac_dist.log_prob(v_a)
            
                # 平均ratio計算
                ratio = (T.exp(log_f - o_f) + T.exp(log_s - o_s) + T.exp(log_v - o_v)) / 3.0
                surr1 = ratio * adv
                surr2 = T.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -T.min(surr1, surr2).mean()
            
                # 熵正則化
                entropy = (dc_dist.entropy() + server_dist.entropy() + ac_dist.entropy()).mean()
                actor_loss -= self.entropy_coef * entropy
            
                # Critic損失
                critic_loss = F.mse_loss(val.squeeze(), ret)
            
                # 總PPO損失
                ppo_loss = actor_loss + 0.5 * critic_loss
            
                # 反向傳播
                ppo_loss.backward()
            
                # 關鍵：屏蔽預測頭的梯度
                for name, param in self.network.named_parameters():
                    if 'pred' in name:
                        param.grad = None
            
                T.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.network.optimizer.step()
            
                total_ppo_loss += ppo_loss.item()
            
                # 記錄損失
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())

        avg_ppo_loss = total_ppo_loss / batch_count if batch_count > 0 else 0.0
        return avg_ppo_loss

    def _synchronized_prediction_update(self, states, dc_actions, server_actions, ac_actions):
        """算法中的預測頭更新部分（使用相同數據）"""
        
        total_pred_loss = 0.0
        
        # 1. 更新 Server → Data Center 預測頭
        server_pred_loss = self._update_single_prediction_head_sync(
            'server_pred_dc', states, dc_actions)
        self.server_pred_dc_losses.append(server_pred_loss)
        total_pred_loss += server_pred_loss
        
        # 2. 更新 AC → Data Center 預測頭  
        ac_pred_dc_loss = self._update_single_prediction_head_sync(
            'ac_pred_dc', states, dc_actions)
        self.ac_pred_dc_losses.append(ac_pred_dc_loss)
        total_pred_loss += ac_pred_dc_loss
        
        # 3. 更新 AC → Server 預測頭
        ac_pred_server_loss = self._update_single_prediction_head_sync(
            'ac_pred_server', states, server_actions)
        self.ac_pred_server_losses.append(ac_pred_server_loss)
        total_pred_loss += ac_pred_server_loss
        
        avg_pred_loss = total_pred_loss / 3.0
        self.prediction_losses.append(avg_pred_loss)
        
        return avg_pred_loss

    def _update_single_prediction_head_sync(self, head_name, states, target_actions):
        """同步更新單個預測頭"""
        self.network.optimizer.zero_grad()
        
        # 前向傳播
        _, pred_dists, _ = self.network(states)
        
        # 根據預測頭類型選擇對應的分布
        if head_name == 'server_pred_dc':
            pred_dist = pred_dists['server_pred_dc']
        elif head_name == 'ac_pred_dc':
            pred_dist = pred_dists['ac_pred_dc']
        elif head_name == 'ac_pred_server':
            pred_dist = pred_dists['ac_pred_server']
        
        # 計算交叉熵損失
        loss = F.cross_entropy(pred_dist.logits, target_actions)
        
        # 反向傳播
        loss.backward()
        
        # 關鍵：只更新對應的預測頭參數
        for name, param in self.network.named_parameters():
            if head_name not in name:
                param.grad = None
        
        T.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.network.optimizer.step()
        
        return loss.item()

    def compute_gae(self, rewards, values, next_values, dones):
        """計算GAE"""
        advantages = T.zeros_like(rewards).to(self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
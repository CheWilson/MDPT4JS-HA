
import random
import numpy as np
import torch as T
import torch.nn.utils
from src.networks.dqn_network import DQNNetwork
from src.data.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN代理類"""
    
    def __init__(self, input_dims, n_actions, lr=0.01, gamma=0.99,
                 epsilon=1.0, eps_min=0.08, batch_size=512, target_update=100,
                 total_tasks_est=80000):
        """
        初始化DQN代理
        
        Args:
            input_dims: 輸入維度
            n_actions: 動作空間大小
            lr: 學習率
            gamma: 折扣因子
            epsilon: 探索率
            eps_min: 最小探索率
            batch_size: 批次大小
            target_update: 目標網絡更新頻率
            total_tasks_est: 總任務數估計（用於計算探索率衰減）
        """
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = (1.0 - eps_min) / (total_tasks_est * 0.5)
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0
        
        # 創建主網絡和目標網絡
        self.Q = DQNNetwork(lr, n_actions, input_dims)
        self.Q_target = DQNNetwork(lr, n_actions, input_dims)
        self.Q_target.load_state_dict(self.Q.state_dict())
        
        # 經驗回放緩衝區
        self.memory = ReplayBuffer()
        
        # 記錄訓練過程
        self.all_losses = []
        self.all_rewards = []
    
    def choose_action(self, state, action_mask=None):
        """
        選擇動作（使用ε-貪婪策略）
        
        Args:
            state: 當前狀態
            action_mask: 動作掩碼（指定哪些動作可選）
            
        Returns:
            int: 選擇的動作
        """
        if np.random.random() > self.epsilon:
            # 利用策略：選擇Q值最高的動作
            if isinstance(state, np.ndarray):
                state = T.tensor(state, dtype=T.float32)
            state = state.to(self.Q.device)
            with T.no_grad():
                actions = self.Q.forward(state)
                if action_mask is not None:
                    action_mask_tensor = T.tensor(action_mask, dtype=T.bool).to(self.Q.device)
                    masked_actions = actions.clone()
                    masked_actions[~action_mask_tensor] = -1e9
                    action = T.argmax(masked_actions).item()
                else:
                    action = T.argmax(actions).item()
        else:
            # 探索策略：隨機選擇動作
            if action_mask is not None and np.any(action_mask):
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
            else:
                action = random.randrange(self.n_actions)
        return action
    
    def learn(self, state, action, reward, next_state):
        """
        更新DQN網路參數
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 轉移到的下一個狀態
        
        Returns:
            float: 損失值
        """
        # 存儲經驗
        self.memory.add(state, action, reward, next_state, 0)
        
        # 如果經驗不足，不進行學習
        if len(self.memory.buffer) < self.batch_size:
            return 0.0
        
        # 清零梯度
        self.Q.optimizer.zero_grad()
        
        # 從經驗回放緩衝區採樣
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
        
        # 將states轉換為張量
        try:
            states_tensor = T.tensor(states, dtype=T.float32).to(self.Q.device)
        except TypeError:
            states_list = []
            for s in states:
                if isinstance(s, np.ndarray):
                    states_list.append(T.tensor(s, dtype=T.float32))
                else:
                    states_list.append(T.tensor(s, dtype=T.float32))
            states_tensor = T.stack(states_list).to(self.Q.device)
        
        # 同樣處理next_states
        try:
            next_states_tensor = T.tensor(next_states, dtype=T.float32).to(self.Q.device)
        except TypeError:
            next_states_list = []
            for s in next_states:
                if isinstance(s, np.ndarray):
                    next_states_list.append(T.tensor(s, dtype=T.float32))
                else:
                    next_states_list.append(T.tensor(s, dtype=T.float32))
            next_states_tensor = T.stack(next_states_list).to(self.Q.device)
        
        actions = T.tensor(actions).to(self.Q.device)
        rewards = T.tensor(rewards).to(self.Q.device)
        
        # 計算當前Q值
        batch_indices = np.arange(len(actions))
        q_pred = self.Q.forward(states_tensor)[batch_indices, actions]
        
        # 計算目標Q值
        with T.no_grad():
            q_next = self.Q_target.forward(next_states_tensor).max(1)[0]
        
        q_target = rewards + self.gamma * q_next
        
        # 計算損失
        loss = self.Q.loss(q_target, q_pred)
        
        # 反向傳播
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q.optimizer.step()
        
        # 更新目標網絡
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
        
        # 更新探索率
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)
        
        # 記錄損失和獎勵
        loss_value = loss.item()
        self.all_losses.append(loss_value)
        self.all_rewards.append(reward)
        
        return loss_value
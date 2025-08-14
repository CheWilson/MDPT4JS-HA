
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn.utils
from src.networks.actor_critic_network import ActorNetwork, CriticNetwork


class ACAgent:
    """Actor-Critic代理類"""
    
    def __init__(self, input_dims, n_actions, lr_actor=0.01, lr_critic=0.01, gamma=0.9):
        """
        初始化AC代理
        
        Args:
            input_dims: 輸入維度
            n_actions: 動作空間大小
            lr_actor: Actor學習率
            lr_critic: Critic學習率
            gamma: 折扣因子
        """
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        
        # 創建Actor和Critic網絡
        self.actor = ActorNetwork(lr_actor, input_dims, n_actions)
        self.critic = CriticNetwork(lr_critic, input_dims)
        
        # 記錄訓練過程
        self.all_losses = []
        self.all_rewards = []
    
    def choose_action(self, state, action_mask=None):
        """
        選擇動作（基於策略概率分布）
        
        Args:
            state: 當前狀態
            action_mask: 動作掩碼（指定哪些動作可選）
            
        Returns:
            int: 選擇的動作
        """
        if isinstance(state, np.ndarray):
            state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        elif state.device != self.actor.device:
            state = state.to(self.actor.device)
        
        with T.no_grad():
            probs = self.actor(state)
            if action_mask is not None:
                action_mask_tensor = T.tensor(action_mask, dtype=T.bool).to(self.actor.device)
                # 獲取原始logits (在softmax之前)
                logits = T.log(probs + 1e-10)  # 反向計算logits
                # 將非法動作的logits設為極小值
                masked_logits = logits.clone()
                masked_logits[~action_mask_tensor] = -1e9
                # 重新應用softmax得到新的概率分布
                masked_probs = F.softmax(masked_logits, dim=-1)
                action = T.multinomial(masked_probs, 1).item()
            else:
                action = T.multinomial(probs, 1).item()
        return action
    
    def learn(self, state, action, reward, next_state):
        """
        更新Actor-Critic網絡參數
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            
        Returns:
            float: 總損失值
        """
        # 轉換為張量
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        next_state = T.tensor(next_state, dtype=T.float32).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        
        # 更新 Critic
        self.critic.optimizer.zero_grad()
        value = self.critic(state)
        value_next = self.critic(next_state).detach()
        td_target = reward + self.gamma * value_next
        critic_loss = F.smooth_l1_loss(value, td_target)
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()
        
        # 更新 Actor
        self.actor.optimizer.zero_grad()
        probs = self.actor(state)
        log_probs = T.log(probs + 1e-10)
        action_log_prob = log_probs[action]
        advantage = (td_target - value).detach()
        
        # actor loss
        actor_loss = -action_log_prob * advantage
        
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor.optimizer.step()
        
        # 記錄損失和獎勵
        loss_value = (critic_loss + actor_loss).item()
        self.all_losses.append(loss_value)
        self.all_rewards.append(reward.item())
        
        return loss_value
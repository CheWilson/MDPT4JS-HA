import torch as T


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    計算 Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: 獎勵張量 [T]
        values: 當前狀態價值張量 [T]  
        next_values: 下一狀態價值張量 [T]
        dones: 結束標誌張量 [T]
        gamma: 折扣因子
        lam: GAE參數λ
        
    Returns:
        tuple: (advantages, returns)
            - advantages: 優勢函數估計 [T]
            - returns: 回報估計 [T]
    """
    T_len = rewards.size(0)
    advantages = T.zeros_like(rewards)
    gae = 0.0
    
    for t in reversed(range(T_len)):
        # 計算TD誤差
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        
        # 更新GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 計算回報
    returns = advantages + values
    
    # 標準化優勢函數
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def compute_returns(rewards, values, dones, gamma=0.99):
    """
    計算蒙特卡羅回報
    
    Args:
        rewards: 獎勵張量 [T]
        values: 狀態價值張量 [T]
        dones: 結束標誌張量 [T]
        gamma: 折扣因子
        
    Returns:
        returns: 回報張量 [T]
    """
    T_len = rewards.size(0)
    returns = T.zeros_like(rewards)
    
    # 從後往前計算回報
    running_return = 0
    for t in reversed(range(T_len)):
        if dones[t]:
            running_return = 0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_td_targets(rewards, values, next_values, dones, gamma=0.99):
    """
    計算TD目標
    
    Args:
        rewards: 獎勵張量 [T]
        values: 當前狀態價值張量 [T]
        next_values: 下一狀態價值張量 [T]
        dones: 結束標誌張量 [T]
        gamma: 折扣因子
        
    Returns:
        td_targets: TD目標張量 [T]
    """
    td_targets = rewards + gamma * next_values * (1 - dones)
    return td_targets


def compute_advantages_simple(rewards, values, next_values, dones, gamma=0.99):
    """
    計算簡單的優勢函數（TD誤差）
    
    Args:
        rewards: 獎勵張量 [T]
        values: 當前狀態價值張量 [T]
        next_values: 下一狀態價值張量 [T]
        dones: 結束標誌張量 [T]
        gamma: 折扣因子
        
    Returns:
        advantages: 優勢函數張量 [T]
    """
    td_targets = compute_td_targets(rewards, values, next_values, dones, gamma)
    advantages = td_targets - values
    
    # 標準化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_n_step_returns(rewards, values, dones, gamma=0.99, n=5):
    """
    計算n步回報
    
    Args:
        rewards: 獎勵張量 [T]
        values: 狀態價值張量 [T]
        dones: 結束標誌張量 [T]
        gamma: 折扣因子
        n: 步數
        
    Returns:
        n_step_returns: n步回報張量 [T]
    """
    T_len = rewards.size(0)
    n_step_returns = T.zeros_like(rewards)
    
    for t in range(T_len):
        n_step_return = 0
        discount = 1
        
        # 計算未來n步的折扣回報
        for k in range(n):
            if t + k >= T_len:
                break
            if dones[t + k] and k > 0:
                break
            n_step_return += discount * rewards[t + k]
            discount *= gamma
        
        # 如果episode沒有結束，加上第n+1步的價值估計
        if t + n < T_len and not dones[t + n - 1]:
            n_step_return += discount * values[t + n]
        
        n_step_returns[t] = n_step_return
    
    return n_step_returns


def normalize_advantages(advantages):
    """
    標準化優勢函數
    
    Args:
        advantages: 優勢函數張量
        
    Returns:
        normalized_advantages: 標準化後的優勢函數張量
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def compute_value_loss(predicted_values, target_values, loss_type='mse'):
    """
    計算價值函數損失
    
    Args:
        predicted_values: 預測的狀態價值
        target_values: 目標狀態價值
        loss_type: 損失類型 ('mse', 'huber', 'smooth_l1')
        
    Returns:
        value_loss: 價值函數損失
    """
    if loss_type == 'mse':
        return T.nn.functional.mse_loss(predicted_values, target_values)
    elif loss_type == 'huber':
        return T.nn.functional.huber_loss(predicted_values, target_values)
    elif loss_type == 'smooth_l1':
        return T.nn.functional.smooth_l1_loss(predicted_values, target_values)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    """
    計算PPO策略損失
    
    Args:
        log_probs: 當前策略的對數概率
        old_log_probs: 舊策略的對數概率
        advantages: 優勢函數
        clip_epsilon: PPO裁剪參數
        
    Returns:
        policy_loss: 策略損失
    """
    # 計算重要性採樣比率
    ratio = T.exp(log_probs - old_log_probs)
    
    # PPO裁剪目標
    surr1 = ratio * advantages
    surr2 = T.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    # 取較小值並求負數（因為我們要最大化）
    policy_loss = -T.min(surr1, surr2).mean()
    
    return policy_loss
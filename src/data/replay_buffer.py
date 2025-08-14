import numpy as np
from collections import deque


class ReplayBuffer:
    """經驗回放緩衝區類"""
    
    def __init__(self, capacity=10000):
        """
        初始化緩衝區
        
        Args:
            capacity: 緩衝區最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        將一筆 transition 加入到 buffer 中
        
        Args:
            state: 當前狀態 (可以是 np.ndarray 或 list)
            action: 所選動作 (int 或 list)
            reward: 獎勵 (float)
            next_state: 下一個狀態
            done: 是否終止 (bool 或 0/1)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        隨機抽取一個 mini-batch
        
        Args:
            batch_size: 批次大小
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in batch]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # 確保所有元素都有相同的形狀
        if all(isinstance(s, np.ndarray) and s.shape == states[0].shape for s in states):
            states = np.array(states, dtype=np.float32)
        else:
            # 如果形狀不一致，保持為列表
            states = list(states)
        
        if all(isinstance(s, np.ndarray) and s.shape == next_states[0].shape for s in next_states):
            next_states = np.array(next_states, dtype=np.float32)
        else:
            next_states = list(next_states)
        
        return (states,
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                next_states,
                np.array(dones, dtype=np.float32))
    
    def dump_all(self):
        """
        取出整個 buffer，並清空它。用於整批訓練
        
        Returns:
            Tuple of lists: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) == 0:
            return [], [], [], [], []
        
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        self.buffer.clear()  # 清空 buffer
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def __len__(self):
        """
        獲取緩衝區中的數據量
        
        Returns:
            int: 數據量
        """
        return len(self.buffer)
    
    def clear(self):
        """清空緩衝區"""
        self.buffer.clear()
    
    def is_ready(self, min_size):
        """
        檢查緩衝區是否準備好進行採樣
        
        Args:
            min_size: 最小數據量要求
            
        Returns:
            bool: 是否準備好
        """
        return len(self.buffer) >= min_size
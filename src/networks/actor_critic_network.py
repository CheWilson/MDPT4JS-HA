import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Actor網絡 - 策略網絡"""
    
    def __init__(self, lr, input_dims, n_actions):
        """
        初始化Actor網絡
        
        Args:
            lr: 學習率
            input_dims: 輸入維度
            n_actions: 動作空間大小
        """
        super(ActorNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def forward(self, state):
        """
        前向傳播
        
        Args:
            state: 輸入狀態
            
        Returns:
            動作概率分布
        """
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        logits = self.fc6(x)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic網絡 - 價值網絡"""
    
    def __init__(self, lr, input_dims):
        """
        初始化Critic網絡
        
        Args:
            lr: 學習率
            input_dims: 輸入維度
        """
        super(CriticNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def forward(self, state):
        """
        前向傳播
        
        Args:
            state: 輸入狀態
            
        Returns:
            狀態價值預測
        """
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        value = self.fc6(x)
        return value
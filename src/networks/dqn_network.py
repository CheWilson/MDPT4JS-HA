import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNNetwork(nn.Module):
    """DQN深度Q網絡"""
    
    def __init__(self, lr, n_actions, input_dims):
        """
        初始化DQN網絡
        
        Args:
            lr: 學習率
            n_actions: 動作空間大小
            input_dims: 輸入維度
        """
        super(DQNNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.to(self.device)
    
    def forward(self, state):
        """
        前向傳播
        
        Args:
            state: 輸入狀態
            
        Returns:
            Q值預測
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
        actions = self.fc6(x)
        return actions
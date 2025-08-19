import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from config import NETWORK_CONFIG


class ACTNetwork(nn.Module):
    """無位置編碼的 Transformer 網路 - 用於ACT4JS"""
    
    def __init__(self, input_dims, n_actions, lr=0.01):
        """
        初始化ACT網絡
        
        Args:
            input_dims: 輸入維度
            n_actions: 動作空間大小
            lr: 學習率
        """
        super(ACTNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.d_model = NETWORK_CONFIG['d_model']
        self.max_seq_len = 128
        
        # 無位置編碼的 Transformer 層
        self.input_layer = nn.Linear(input_dims, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=NETWORK_CONFIG['nhead'], 
            dim_feedforward=NETWORK_CONFIG['dim_feedforward'], 
            dropout=NETWORK_CONFIG['dropout'], 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=NETWORK_CONFIG['num_layers']
        )
        
        # 特徵層
        self.shared_features = nn.Linear(self.d_model, 512)
        
        # 動作頭
        self.action_head = nn.Linear(512, n_actions)
        
        # 價值函數頭
        self.value_head = nn.Linear(512, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def _process_input(self, state):
        """處理輸入狀態，沒有位置編碼"""
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        if state.dim() == 2:
            state = state.unsqueeze(0)
               
        x = F.relu(self.input_layer(state))
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 對序列維度取平均
        
        return F.relu(self.shared_features(x))
    
    def forward(self, state):
        """前向傳播返回動作分布和一個值函數"""
        shared_features = self._process_input(state)
        
        # 生成動作分布
        action_logits = self.action_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 創建分布
        action_dist = Categorical(action_probs)
        
        # 計算值函數
        value = self.value_head(shared_features)
        
        return action_dist, value
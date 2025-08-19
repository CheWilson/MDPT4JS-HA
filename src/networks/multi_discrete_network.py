import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from config import NETWORK_CONFIG


class MultiDiscreteACTNetwork(nn.Module):
    """MDPT4JS 的多離散動作網路 - 有位置編碼，使用完整狀態"""
    
    def __init__(self, input_dims, n_actions_list, lr=0.01, total_acs=1000):
        """
        初始化多離散ACT網絡
        
        Args:
            input_dims: 每個AC的特徵維度
            n_actions_list: 各層動作數量列表
            lr: 學習率
            total_acs: 總AC數量
        """
        super(MultiDiscreteACTNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.d_model = NETWORK_CONFIG['d_model']
        self.total_acs = total_acs
        self.n_actions_list = n_actions_list
        
        # 共享層
        self.input_layer = nn.Linear(input_dims, self.d_model)
        
        # 改進的位置編碼 - 基於真實拓撲結構
        self.position_encoder = self._create_enhanced_positional_encoding(self.total_acs, self.d_model)
        
        # Transformer 編碼器
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
        
        # 共享特徵層
        self.shared_features = nn.Linear(self.d_model, 128)
        
        # 多維動作頭
        self.dc_head = nn.Linear(128, n_actions_list[0])
        self.server_head = nn.Linear(128, n_actions_list[1])
        self.ac_head = nn.Linear(128, n_actions_list[2])
        
        # 價值函數頭
        self.value_head = nn.Linear(128, 1)
        
        # 權重初始化
        self._initialize_weights()
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def _create_enhanced_positional_encoding(self, max_seq_len, d_model):
        """
        創建增強的位置編碼，考慮資料中心的階層結構
        
        Args:
            max_seq_len: 最大序列長度（總AC數量）
            d_model: 模型維度
        
        Returns:
            位置編碼張量 [max_seq_len, d_model]
        """
        pe = T.zeros(max_seq_len, d_model)
        
        # 環境參數（根據您的設定）
        dc_size = 50
        ac_per_server = 2
        
        for ac_idx in range(max_seq_len):
            # 計算該AC在拓撲中的位置
            dc_id = ac_idx // (dc_size * ac_per_server)
            remaining = ac_idx % (dc_size * ac_per_server)
            server_id = remaining // ac_per_server
            ac_id = remaining % ac_per_server
            
            # 為不同的層級創建不同頻率的位置編碼
            for i in range(d_model):
                if i % 4 == 0:  # DC level encoding
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.sin(dc_id * div_term)
                elif i % 4 == 1:  # Server level encoding
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.cos(server_id * div_term)
                elif i % 4 == 2:  # AC level encoding
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.sin(ac_id * div_term)
                else:  # Combined encoding
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    combined_pos = ac_idx
                    pe[ac_idx, i] = math.cos(combined_pos * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        return pe.to(self.device)
    
    def _initialize_weights(self):
        """初始化網絡權重以提高穩定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 對於動作頭使用較小的初始化值
        nn.init.xavier_uniform_(self.dc_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.server_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.ac_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value_head.weight, gain=0.01)
    
    def _process_input(self, state):
        """處理輸入狀態，使用完整狀態並添加增強的位置編碼"""
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # 檢查輸入是否包含NaN值
        if T.isnan(state).any():
            state = T.where(T.isnan(state), T.zeros_like(state), state)
        
        if T.isinf(state).any():
            state = T.where(T.isinf(state), T.ones_like(state) * 1e6, state)
        
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        batch_size, num_acs, feature_dim = state.shape

        # 直接處理完整狀態
        x = F.relu(self.input_layer(state))
        
        # 添加位置編碼（只取實際使用的長度）
        x = x + self.position_encoder[:, :num_acs, :]
        
        # Transformer 處理
        x = self.transformer_encoder(x)
        
        # 對序列維度取平均
        x = x.mean(dim=1)
        
        features = F.relu(self.shared_features(x))
        
        # 檢查特徵中的NaN值
        if T.isnan(features).any():
            features = T.where(T.isnan(features), T.zeros_like(features), features)
        
        return features
    
    def forward(self, state):
        """前向傳播返回三個動作分布和一個值函數，增強數值穩定性"""
        shared_features = self._process_input(state)
        
        # 生成三個動作分布
        dc_logits = self.dc_head(shared_features)
        server_logits = self.server_head(shared_features)
        ac_logits = self.ac_head(shared_features)
        
        # 使用穩定的softmax
        temperature = 1.0
        dc_probs = F.softmax(dc_logits / temperature, dim=-1)
        server_probs = F.softmax(server_logits / temperature, dim=-1)
        ac_probs = F.softmax(ac_logits / temperature, dim=-1)
        
        # 創建分布
        dc_dist = Categorical(dc_probs)
        server_dist = Categorical(server_probs)
        ac_dist = Categorical(ac_probs)
        
        # 計算值函數
        value = self.value_head(shared_features)
        
        if T.isnan(value).any():
            value = T.where(T.isnan(value), T.zeros_like(value), value)
        
        return [dc_dist, server_dist, ac_dist], value


class MultiDiscreteACT4JSPredNetwork(nn.Module):
    """MDPT4JS-HA 的網路 - 有位置編碼和預測頭，使用完整狀態"""
    
    def __init__(self, input_dims, n_actions_list, lr=0.01, total_acs=1000):
        """
        初始化MDPT4JS-HA網絡
        
        Args:
            input_dims: 輸入維度
            n_actions_list: 動作列表
            lr: 學習率
            total_acs: 總AC數量
        """
        super(MultiDiscreteACT4JSPredNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.d_model = NETWORK_CONFIG['d_model']
        self.total_acs = total_acs
        self.n_actions_list = n_actions_list
        
        # 共享層
        self.input_layer = nn.Linear(input_dims, self.d_model)
        
        # 使用與 MultiDiscreteACTNetwork 相同的增強位置編碼
        self.position_encoder = self._create_enhanced_positional_encoding(self.total_acs, self.d_model)
        
        # Transformer 編碼器
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
        
        # 共享特徵層
        self.shared_features = nn.Linear(self.d_model, 128)
        
        # 多維動作頭
        self.dc_head = nn.Linear(128, n_actions_list[0])
        self.server_head = nn.Linear(128, n_actions_list[1])
        self.ac_head = nn.Linear(128, n_actions_list[2])
        
        # 預測頭
        self.server_pred_dc_head = nn.Linear(128, n_actions_list[0])
        self.ac_pred_dc_head = nn.Linear(128, n_actions_list[0])
        self.ac_pred_server_head = nn.Linear(128, n_actions_list[1])
        
        # 價值函數頭
        self.value_head = nn.Linear(128, 1)
        
        # 權重初始化
        self._initialize_weights()
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def _create_enhanced_positional_encoding(self, max_seq_len, d_model):
        """與 MultiDiscreteACTNetwork 相同的增強位置編碼"""
        pe = T.zeros(max_seq_len, d_model)
        
        dc_size = 50
        ac_per_server = 2
        
        for ac_idx in range(max_seq_len):
            dc_id = ac_idx // (dc_size * ac_per_server)
            remaining = ac_idx % (dc_size * ac_per_server)
            server_id = remaining // ac_per_server
            ac_id = remaining % ac_per_server
            
            for i in range(d_model):
                if i % 4 == 0:  # DC level
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.sin(dc_id * div_term)
                elif i % 4 == 1:  # Server level
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.cos(server_id * div_term)
                elif i % 4 == 2:  # AC level
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.sin(ac_id * div_term)
                else:  # Combined
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                    pe[ac_idx, i] = math.cos(ac_idx * div_term)
        
        pe = pe.unsqueeze(0)
        return pe.to(self.device)
    
    def _initialize_weights(self):
        """初始化網絡權重以提高穩定性和性能"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 對所有頭使用較小的初始化值
        heads = [self.dc_head, self.server_head, self.ac_head,
                 self.server_pred_dc_head, self.ac_pred_dc_head, 
                 self.ac_pred_server_head, self.value_head]
        for head in heads:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
    
    def _process_input(self, state):
        """處理輸入狀態，使用完整狀態並添加增強的位置編碼"""
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        _, num_acs, _ = state.shape
        
        x = F.relu(self.input_layer(state))
        
        # 添加增強的位置編碼
        x = x + self.position_encoder[:, :num_acs, :]
        
        # Transformer 處理
        x = self.transformer_encoder(x)
        
        # 對序列維度取平均
        x = x.mean(dim=1)
        
        return F.relu(self.shared_features(x))
    
    def forward(self, state):
        """前向傳播返回動作分布、預測頭分布和值函數"""
        shared_features = self._process_input(state)
        
        # 生成三個動作分布
        dc_logits = self.dc_head(shared_features)
        server_logits = self.server_head(shared_features)
        ac_logits = self.ac_head(shared_features)
        
        # 使用穩定的softmax
        dc_probs = F.softmax(dc_logits, dim=-1)
        server_probs = F.softmax(server_logits, dim=-1)
        ac_probs = F.softmax(ac_logits, dim=-1)    
        
        # 創建分布
        dc_dist = Categorical(dc_probs)
        server_dist = Categorical(server_probs)
        ac_dist = Categorical(ac_probs)
        
        # 生成預測頭的輸出
        server_pred_dc_logits = self.server_pred_dc_head(shared_features)
        ac_pred_dc_logits = self.ac_pred_dc_head(shared_features)
        ac_pred_server_logits = self.ac_pred_server_head(shared_features)
        
        # 使用穩定的softmax
        server_pred_dc_probs = F.softmax(server_pred_dc_logits, dim=-1)
        ac_pred_dc_probs = F.softmax(ac_pred_dc_logits, dim=-1)
        ac_pred_server_probs = F.softmax(ac_pred_server_logits, dim=-1)
        
        # 創建預測頭分布
        server_pred_dc_dist = Categorical(server_pred_dc_probs)
        ac_pred_dc_dist = Categorical(ac_pred_dc_probs)
        ac_pred_server_dist = Categorical(ac_pred_server_probs)
        
        # 將預測頭分布整合到字典中
        pred_dists = {
            'server_pred_dc': server_pred_dc_dist,
            'ac_pred_dc': ac_pred_dc_dist,
            'ac_pred_server': ac_pred_server_dist
        }
        
        # 計算值函數
        value = self.value_head(shared_features)
        
        return [dc_dist, server_dist, ac_dist], pred_dists, value
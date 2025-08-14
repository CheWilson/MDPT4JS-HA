import random
import numpy as np
import torch
import os


def set_reproducibility(seed=42):
    """設置所有隨機種子以確保結果可復現"""
    
    # 設置Python內建random模組的種子
    random.seed(seed)
    
    # 設置NumPy的種子
    np.random.seed(seed)
    
    # 設置PyTorch的種子
    torch.manual_seed(seed)
    
    # 如果使用CUDA，設置CUDA的種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        
        # 這兩行是為了確保CUDA的運算是確定性的
        # 可能會影響性能，但對復現性很重要
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 設置Python的hash seed（影響字典和集合的順序）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已設置所有隨機種子為{seed}，確保結果可復現")
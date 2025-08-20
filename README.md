# Deep Reinforcement Learning for Cloud Task Scheduling

[![Python](https://img.shields.io/badge/Python-3.11.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive deep reinforcement learning framework for cloud task scheduling that compares 7 algorithms including transformer-based methods (MDPT4JS, ACT4JS) with traditional approaches (DQN, Actor-Critic) and baseline schedulers.

## 1. Environment Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python**: 3.11.7 (recommended) or 3.10+
- **PyTorch**: 2.5.1+cu118 (GPU) or 2.5.1 (CPU)
- **CUDA**: 11.8 (for GPU acceleration)
- **Memory**: 8GB RAM minimum, 16GB recommended

> **Note**: Training results may vary across different hardware configurations and GPU types.

## 2. Installation

###  Clone this project
```bash
git clone https://github.com/CheWilson/MDPT4JS-HA.git
cd MDPT4JS-HA
```
### Create New Conda Environment
```bash
# Create environment with Python 3.11.7
conda create -n rl_scheduling_1 python=3.11.7 -y

# Activate environment
conda activate rl_scheduling

# Install PyTorch with CUDA 11.8 (recommended)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

## 3. Repository Structure
```bash
rl_task_scheduling/
├── main.py                         #  主程式入口 - 執行完整比較實驗
├── config.py                       #  全域配置文件
├── requirements.txt                 #  依賴包列表
├── README.md                        #  項目文檔
│
├── src/                            #  核心模組
│   ├── __init__.py
│   ├── environment/                #  雲端環境模擬
│   │   ├── __init__.py
│   │   ├── environment.py          # 三層資源架構 (DC→Server→AC)
│   │   ├── virtual_clock.py        # 虛擬時間管理
│   │   └── job_dag.py              # 作業和任務DAG管理
│   │
│   ├── agents/                     #  強化學習代理
│   │   ├── __init__.py
│   │   ├── dqn_agent.py           # Deep Q-Network 代理
│   │   ├── ac_agent.py            # Actor-Critic 代理
│   │   ├── act4js_agent.py        # ACT4JS Transformer代理
│   │   ├── mdpt4js_agent.py       # MDPT4JS 共享Transformer代理
│   │   └── mdpt4js_ha_agent.py    # MDPT4JS-HA 帶預測頭代理
│   │
│   ├── networks/                   #  神經網絡架構
│   │   ├── __init__.py
│   │   ├── dqn_network.py         # DQN 網絡
│   │   ├── actor_critic_network.py # AC 網絡
│   │   ├── transformer_network.py  # Transformer 網絡
│   │   └── multi_discrete_network.py # 多離散動作網絡
│   │
│   ├── schedulers/                 #  基準調度算法
│   │   ├── __init__.py
│   │   ├── base_scheduler.py      # 調度器基類
│   │   ├── first_fit.py           # First Fit 調度器
│   │   └── random_fit.py          # Random Fit 調度器
│   │
│   ├── data/                       #  數據管理
│   │   ├── __init__.py
│   │   ├── job_manager.py         # 作業數據管理器
│   │   └── replay_buffer.py       # 經驗回放緩衝區
│   │
│   ├── training/                   #  訓練模組
│   │   ├── __init__.py
│   │   ├── trainer.py             # 統一訓練器
│   │   └── gae_utils.py           # GAE計算工具
│   │
│   └── evaluation/                 #  評估和可視化
│       ├── __init__.py
│       ├── visualization.py       # 靜態圖表生成
│       └── dynamic_visualization.py # 動態GIF生成
│
├── utils/                          #  工具函數
│   ├── __init__.py
│   ├── reproducibility.py         # 隨機種子設置
│   └── model_utils.py             # 模型保存/加載工具
│
├── experiments/                    #  實驗腳本
│   ├── __init__.py
│   ├── run_comparison.py          # 方法比較實驗
│   ├── run_single_method.py       # 單一方法訓練
│   ├── model_testing.py           # 模型測試
│   └── generate_gif_visualization.py # GIF 生成
│
├── data/                           #  數據文件目錄
│   └── raw/
│       ├── batch_task.csv         # 訓練數據集
│       └── test.csv               # 測試數據集
│
├── models/                         # 按容器配置分組的訓練模型
│   ├── ac2/                       # AC=2配置 (1000總容器)
│   │   ├── dqn/
│   │   │   ├── dc_Q_ep100.pth
│   │   │   ├── dc_Q_target_ep100.pth
│   │   │   ├── server_Q_ep100.pth
│   │   │   ├── server_Q_target_ep100.pth
│   │   │   ├── ac_Q_ep100.pth
│   │   │   └── ac_Q_target_ep100.pth
│   │   ├── ac/
│   │   │   ├── dc_actor_ep100.pth
│   │   │   ├── dc_critic_ep100.pth
│   │   │   ├── server_actor_ep100.pth
│   │   │   ├── server_critic_ep100.pth
│   │   │   ├── ac_actor_ep100.pth
│   │   │   └── ac_critic_ep100.pth
│   │   ├── act4js/
│   │   │   ├── dc_network_ep100.pth
│   │   │   ├── server_network_ep100.pth
│   │   │   └── ac_network_ep100.pth
│   │   ├── mdpt4js/
│   │   │   └── network_ep100.pth
│   │   └── mdpt4js-ha/
│   │       └── network_ep100.pth
│   ├── ac6/                       # AC = 6 配置 (3000總容器)
│   └── ac10/                      # AC = 10 配置 (5000總容器)
│
└── results/                        #  按實驗配置分組的結果
    ├── ac2/                       # AC=2 配置的實驗結果
    │   ├── plots/                 #  靜態圖表
    │   │   ├── batch_power_consumption_comparison_original.png
    │   │   ├── batch_power_consumption_comparison_smoothed.png
    │   │   ├── training_losses_comparison.png
    │   │   ├── episode_rewards_comparison.png
    │   │   ├── success_rates_comparison.png
    │   │   ├── mdpt4js_ha_detailed_losses.png
    │   │   └── act4js_detailed_losses.png
    │   ├── metrics/               #  性能指標數據
    │   │   ├── rewards_ac2_dqn.npy
    │   │   ├── power_ac2_dqn.npy
    │   │   ├── losses_ac2_dqn.npy
    │   │   ├── batch_power_ac2_dqn.npy
    │   │   └── [其他方法的相同格式文件]
    │   └── logs/                  ]
    │       ├── ac2_training.log
    │       └── ac2_experiment_summary.txt
    ├── ac6/                       # AC = 6 配置的實驗結果
    ├── ac10/                      # AC = 10 配置的實驗結果
    └── visualization_results/      # 動態可視化結果
        ├── gifs/                  # GIF動畫文件
        │   ├── mdpt4js_ha_ac2_100jobs.gif
        │   ├── mdpt4js_ac2_100jobs.gif
        │   ├── act4js_ac2_100jobs.gif
        │   ├── dqn_ac2_100jobs.gif
        │   ├── ac_ac2_100jobs.gif
        │   ├── first_fit_ac2_100jobs.gif
        │   └── random_fit_ac2_100jobs.gif
        └── visualization_summary.csv
    
```

## 4. Supported Algorithms

### 4.1 Baseline Methods

- Random Fit: Randomly selects available resources for task allocation
- First Fit: Sequentially finds the first available resource slot

### 4.2 Deep Reinforcement Learning Methods

- DQN: Deep Q-Network with hierarchical action selection
- AC: Actor-Critic with separate policy and value networks
- ACT4JS: Transformer-based architecture with three independent networks
- MDPT4JS: Multi-discrete action transformer with shared encoder
- MDPT4JS-HA: Hierarchical attention variant with prediction heads

## 5. Usage Examples

### 5.1 Quick Start - Complete Comparison
```bash
# Run all algorithms

python main.py --servers 500 --dc-size 50 --ac-configs 2 --episodes 100 --jobs 100 --csv ./data/raw/batch_task.csv

# Generated Files:
results/
├── ac2/plots/         # All comparison charts for AC=2 
├── ac6/plots/         # All comparison charts for AC=6 
└── ac10/plots/        # All comparison charts for AC=10 
models/
├── ac2/[method]/      # Trained models for AC=2 
├── ac6/[method]/      # Trained models for AC=6 
└── ac10/[method]/     # Trained models for AC=10 
```

### 5.2 Single Algorithm Training
```bash
# Train DQN 
python experiments/run_single_method.py DQN --servers 500 --dc-size 50 --ac-per-server 2 --episodes  100 --jobs 100 --csv ./data/raw/batch_task.csv

# Train MDPT4JS-HA
python experiments/run_single_method.py MDPT4JS-HA --servers 500 --dc-size 50 --ac-per-server 2 --episodes 100 --jobs 100 --csv ./data/raw/batch_task.csv

```

### 5.3 Model Testing
```bash
# Test all saved models
python experiments/model_testing.py --servers 500 --dc-size 50 --ac-configs 2 6 10 --episodes 100 --jobs 100 --csv ./data/raw/test.csv

# Generated Files:
test_results/
├── test_results.npy                   # All test results
├── rejection_rates.npy                # Rejection rate data
├── simulation_times.npy               # Simulation time data
├── model_testing_summary.csv          # Test results summary
└── container_config_comparison.png    # Container configuration comparison 
```

## 6. Configuration
Modify config.py for different experiment settings:
```python
# Quick testing configuration
TRAINING_CONFIG = {
    'num_episodes': 100,
    'jobs_per_episode': 100,
    'random_seed': 42,
}

ENV_CONFIG = {
    'total_servers': 500,
    'dc_size': 50,
    'ac_per_server': 2,
}
```

## 7.  Generate GIF Animations for All Methods
```bash
# Generate dynamic visualization for all methods
python experiments/generate_gif_visualization.py --model-path ./models  --output-path ./visualization_results --ac-config 2 --csv ./data/raw/batch_task.csv

- --model-path ./models: Location to load trained model parameters
- --output-path ./visualization_results: GIF storage location
- --ac-config 2: Visualization based on 2 containers per server configuration
- --csv ./data/raw/batch_task.csv: Data source for animation generation

# Generated GIF Files:

visualization_results/gifs/
├── mdpt4js_ha_ac2_100jobs.gif        # MDPT4JS-HA scheduling process animation
├── mdpt4js_ac2_100jobs.gif           # MDPT4JS scheduling process animation
├── act4js_ac2_100jobs.gif            # ACT4JS scheduling process animation
├── dqn_ac2_100jobs.gif               # DQN scheduling process animation
├── ac_ac2_100jobs.gif                # AC scheduling process animation
├── first_fit_ac2_100jobs.gif         # First Fit scheduling process animation
└── random_fit_ac2_100jobs.gif        # Random Fit scheduling process animation

```












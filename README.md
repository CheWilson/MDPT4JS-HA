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

### Create New Conda Environment
```bash
# Create environment with Python 3.11.7
conda create -n rl_scheduling python=3.11.7 -y

# Activate environment
conda activate rl_scheduling

# Install PyTorch with CUDA 11.8 (recommended)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install numpy pandas matplotlib seaborn plotly tqdm psutil openpyxl colorama sympy networkx
```

## 3. Repository Structure
```bash
rl_task_scheduling/
│
├── main.py                     # Main entry point for comprehensive experiments
├── config.py                   # Configuration file for all parameters
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── src/                        # Core source code
│   ├── __init__.py             # Package initialization
│   │
│   ├── environment/            # Task scheduling environment
│   │   ├── __init__.py         # Package initialization
│   │   ├── environment.py      # Main Environment class
│   │   ├── virtual_clock.py    # Virtual time management
│   │   └── job_dag.py          # Job, Task, and DAG classes
│   │
│   ├── agents/                 # Reinforcement learning agents
│   │   ├── __init__.py         # Package initialization
│   │   ├── dqn_agent.py        # Deep Q-Network agent
│   │   ├── ac_agent.py         # Actor-Critic agent
│   │   ├── act4js_agent.py     # ACT4JS transformer agent
│   │   ├── mdpt4js_agent.py    # Multi-discrete transformer agent
│   │   └── mdpt4js_ha_agent.py # Hierarchical attention agent
│   │
│   ├── networks/               # Neural network architectures
│   │   ├── __init__.py         # Package initialization
│   │   ├── dqn_network.py      # DQN network
│   │   ├── actor_critic_network.py # Actor-Critic networks
│   │   ├── transformer_network.py  # Transformer architecture
│   │   └── multi_discrete_network.py # Multi-discrete action networks
│   │
│   ├── schedulers/             # Baseline scheduling algorithms
│   │   ├── __init__.py         # Package initialization
│   │   ├── base_scheduler.py   # Abstract scheduler base class
│   │   ├── first_fit.py        # First Fit scheduler
│   │   └── random_fit.py       # Random Fit scheduler
│   │
│   ├── data/                   # Data management
│   │   ├── __init__.py         # Package initialization
│   │   ├── job_manager.py      # Job data loading and management
│   │   └── replay_buffer.py    # Experience replay buffer
│   │
│   ├── training/               # Training utilities
│   │   ├── __init__.py         # Package initialization
│   │   ├── trainer.py          # Main training orchestrator
│   │   └── gae_utils.py        # Generalized Advantage Estimation
│   │
│   └── evaluation/             # Evaluation and visualization
│       ├── __init__.py         # Package initialization
│       └── visualization.py    # Plotting and analysis tools
│
├── utils/                      # Utility functions
│   ├── __init__.py             # Package initialization
│   ├── reproducibility.py     # Random seed management
│   └── model_utils.py          # Model save/load utilities
│
├── experiments/                # Experiment scripts
│   ├── __init__.py             # Package initialization
│   ├── run_comparison.py       # Multi-algorithm comparison
│   ├── run_single_method.py    # Single algorithm training
│   └── model_testing.py        # Model evaluation and testing
│
├── data/                       # Data files
│   └── raw/                    # Raw task data (CSV files)
│
├── models/                     # Saved model weights
│   ├── dqn/                    # DQN model checkpoints
│   ├── ac/                     # Actor-Critic models
│   ├── act4js/                 # ACT4JS models
│   ├── mdpt4js/                # MDPT4JS models
│   └── mdpt4js-ha/             # MDPT4JS-HA models
│
├── results/                    # Experiment results
│   ├── plots/                  # Generated figures
│   ├── metrics/                # Numerical results
│   └── logs/                   # Training logs
│
└── tests/                      # Unit tests
    ├── __init__.py             # Package initialization
    ├── test_environment.py     # Environment tests
    └── test_agents.py          # Agent tests
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

python main.py --servers 500 --dc-size 10 --ac-per-server 2 --episodes 100 --jobs 100
```

### 5.2 Single Algorithm Training
```bash
# Train DQN 
python experiments/run_single_method.py DQN --servers 500 --dc-size 10 --ac-per-server 2 --episodes 100 --jobs 100

# Train MDPT4JS-HA
python experiments/run_single_method.py MDPT4JS-HA --servers 500 --dc-size 10 --ac-per-server 2 --episodes 100 --jobs 100

```

### 5.3 Algorithm Comparison
```bash
# Default comparison
python experiments/run_comparison.py

# Custom parameters
python experiments/run_comparison.py --episodes 50 --jobs 100
```

### 5.4 Model Testing
```bash
# Test all saved models
python experiments/model_testing.py
```

## 6. Configuration
Modify config.py for different experiment settings:
```python
# Quick testing configuration
TRAINING_CONFIG = {
    'num_episodes': 10,
    'jobs_per_episode': 20,
    'random_seed': 42,
}

ENV_CONFIG = {
    'total_servers': 100,
    'dc_size': 10,
    'ac_per_server': 2,
}
```
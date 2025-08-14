# 環境參數
ENV_CONFIG = {
    'total_servers': 500,
    'dc_size': 50,
    'ac_per_server': 2,
    'max_seq_len': 128,
}

# 訓練參數
TRAINING_CONFIG = {
    'num_episodes': 100,
    'jobs_per_episode': 100,
    'batch_size': 100,  # 每批次的job數量
    'random_seed': 42,
}

# DQN 參數
DQN_CONFIG = {
    'lr': 0.01,
    'gamma': 0.99,
    'epsilon': 1.0,
    'eps_min': 0.08,
    'batch_size': 512,
    'target_update': 100,
    'total_tasks_est': 80000,
}

# Actor-Critic 參數
AC_CONFIG = {
    'lr_actor': 0.0001,
    'lr_critic': 0.0001,
    'gamma': 0.9,
}

# ACT4JS 參數
ACT4JS_CONFIG = {
    'lr': 0.01,
    'gamma': 0.9,
    'clip_epsilon': 0.2,
    'buffer_capacity': 10000,
    'buffer_size': 512,
    'mini_batch_size': 32,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'entropy_coef': 1e-2,
}

# MDPT4JS 參數
MDPT4JS_CONFIG = {
    'lr': 0.001,
    'gamma': 0.9,
    'clip_epsilon': 0.2,
    'buffer_capacity': 10000,
    'buffer_size': 256,
    'mini_batch_size': 64,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'entropy_coef': 1e-2,
}

# MDPT4JS-HA 參數
MDPT4JS_HA_CONFIG = {
    'lr': 1e-4,
    'gamma': 0.9,
    'clip_epsilon': 0.2,
    'buffer_capacity': 10000,
    'buffer_size': 256,
    'mini_batch_size': 32,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'entropy_coef': 1e-2,
}

# 網絡參數
NETWORK_CONFIG = {
    'd_model': 256,
    'nhead': 2,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'num_layers': 1,
}

# 路徑配置
PATHS = {
    'save_path': "./models",
    'data_path': "./data",
    'results_path': "./results",
    'logs_path': "./results/logs",
    'plots_path': "./results/plots",
}

# 顏色配置（用於繪圖）
METHOD_COLORS = {
    'Random Fit': '#1f77b4',
    'First Fit': '#ff7f0e',
    'DQN': '#2ca02c',
    'AC': '#d62728',
    'ACT4JS': '#9467bd',
    'MDPT4JS': '#8c564b',
    'MDPT4JS-HA': '#e377c2'
}

# 任務參數
TASK_CONFIG = {
    'runtime_min': 25,
    'runtime_max': 35,
    'deadline_buffer': 1000,
    'cpu_range': (0.01, 0.1),
    'ram_range': (0.01, 0.1),
    'disk_range': (0.01, 0.1),
    'dependency_prob': 0.7,
}

# 電力消耗參數
POWER_CONFIG = {
    'static_power': 0.8,
    'alpha': 0.5,
    'beta': 50,
    'utilization_threshold': 0.7,
    'price_threshold': 1.5,
    'low_price': 5.91,
    'high_price': 8.27,
}

# 獎勵參數
REWARD_CONFIG = {
    'power_penalty_factor': 10,
    'rejection_penalty': -5,
}

# 視覺化參數
PLOT_CONFIG = {
    'figsize': (10, 6),
    'linewidth': 2.5,
    'dpi': 200,
    'smoothing_window': 5,
    'large_smoothing_window': 10,
}
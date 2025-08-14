import os
import sys
import numpy as np

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.reproducibility import set_reproducibility
from utils.model_utils import create_model_directories
from src.environment.environment import Environment
from src.data.job_manager import JobManager
from src.schedulers.first_fit import FirstFitScheduler
from src.schedulers.random_fit import RandomFitScheduler
from src.training.trainer import Trainer
from src.evaluation.visualization import (
    plot_batch_power_consumption, 
    plot_training_losses,
    plot_episode_rewards,
    plot_success_rates,
    create_summary_report
)


def run_full_comparison(csv_file_path=None, num_episodes=50, jobs_per_episode=100):
    """
    運行完整的方法比較實驗
    
    Args:
        csv_file_path: CSV數據文件路徑
        num_episodes: 訓練episode數量
        jobs_per_episode: 每個episode的job數量
    
    Returns:
        dict: 實驗結果字典
    """
    print("=" * 80)
    print("雲端任務調度強化學習方法比較實驗")
    print("=" * 80)
    
    # 設置隨機種子
    set_reproducibility(TRAINING_CONFIG['random_seed'])
    
    # 創建必要目錄
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 設置作業管理器
    job_manager = JobManager()
    if csv_file_path and os.path.exists(csv_file_path):
        print(f"載入CSV數據: {csv_file_path}")
        job_manager.load_all_jobs_from_csv(csv_file_path)
    else:
        print("使用隨機生成數據")
        csv_file_path = None
    
    # 建立環境
    env_config = ENV_CONFIG.copy()
    environments = {}
    methods = ['Random Fit', 'First Fit', 'DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
    
    for method in methods:
        environments[method] = Environment(
            total_servers=env_config['total_servers'],
            dc_size=env_config['dc_size'],
            ac_per_server=env_config['ac_per_server']
        )
    
    # 創建模型保存目錄
    create_model_directories(PATHS['save_path'], methods)
    
    # 存儲結果
    all_results = {}
    
    print("\n" + "=" * 60)
    print("執行基準方法")
    print("=" * 60)
    
    # 1. Random Fit
    print(f"\n執行 Random Fit ({num_episodes} episodes)...")
    random_scheduler = RandomFitScheduler(environments['Random Fit'])
    random_rewards, random_power = random_scheduler.run_multiple_episodes(
        num_episodes=num_episodes,
        jobs_per_episode=jobs_per_episode,
        csv_file_path=csv_file_path,
        job_manager=job_manager
    )
    all_results['Random Fit'] = {
        'rewards': random_rewards,
        'power': random_power,
        'env': environments['Random Fit']
    }
    
    # 2. First Fit
    print(f"\n執行 First Fit ({num_episodes} episodes)...")
    first_scheduler = FirstFitScheduler(environments['First Fit'])
    first_rewards, first_power = first_scheduler.run_multiple_episodes(
        num_episodes=num_episodes,
        jobs_per_episode=jobs_per_episode,
        csv_file_path=csv_file_path,
        job_manager=job_manager
    )
    all_results['First Fit'] = {
        'rewards': first_rewards,
        'power': first_power,
        'env': environments['First Fit']
    }
    
    print("\n" + "=" * 60)
    print("執行強化學習方法")
    print("=" * 60)
    
    # 3. 強化學習方法
    trainer = Trainer()
    rl_methods = ['DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
    
    for method in rl_methods:
        print(f"\n執行 {method} ({num_episodes} episodes)...")
        
        rewards, losses, agents = trainer.train_method(
            env=environments[method],
            method=method,
            num_episodes=num_episodes,
            jobs_per_episode=jobs_per_episode,
            save_path=PATHS['save_path'],
            csv_file_path=csv_file_path,
            job_manager=job_manager
        )
        
        all_results[method] = {
            'rewards': rewards,
            'losses': losses,
            'agents': agents,
            'env': environments[method]
        }
    
    print("\n" + "=" * 60)
    print("保存實驗結果")
    print("=" * 60)
    
    # 保存數值數據
    for method, result in all_results.items():
        method_safe = method.lower().replace(" ", "_")
        
        # 保存獎勵數據
        np.save(f'{PATHS["results_path"]}/rewards_{method_safe}.npy', 
                np.array(result['rewards']))
        
        # 保存能耗數據
        if 'power' in result:
            np.save(f'{PATHS["results_path"]}/power_{method_safe}.npy', 
                    np.array(result['power']))
        
        # 保存損失數據
        if 'losses' in result:
            np.save(f'{PATHS["results_path"]}/losses_{method_safe}.npy', 
                    np.array(result['losses']))
        
        # 保存批次能耗數據
        if result['env'].batch_power_records:
            batch_power_data = np.array([sum(batch) for batch in result['env'].batch_power_records])
            np.save(f'{PATHS["results_path"]}/batch_power_{method_safe}.npy', batch_power_data)
    
    print("\n" + "=" * 60)
    print("生成可視化圖表")
    print("=" * 60)
    
    # 創建環境字典用於可視化
    envs_for_plot = {method: result['env'] for method, result in all_results.items()}
    
    # 繪製批次能耗比較圖
    plot_batch_power_consumption(envs_for_plot, methods, PATHS['plots_path'])
    
    # 繪製獎勵比較圖（僅RL方法）
    rl_rewards = {method: result['rewards'] for method, result in all_results.items() 
                  if method in rl_methods}
    if rl_rewards:
        plot_episode_rewards(rl_rewards, rl_methods, PATHS['plots_path'])
    
    # 繪製損失比較圖（僅RL方法）
    rl_losses = {method: result['losses'] for method, result in all_results.items() 
                 if 'losses' in result}
    if rl_losses:
        plot_training_losses(rl_losses, list(rl_losses.keys()), PATHS['plots_path'])
    
    # 繪製成功率比較圖
    plot_success_rates(envs_for_plot, methods, PATHS['plots_path'])
    
    # 生成總結報告
    create_summary_report(envs_for_plot, methods, PATHS['results_path'])
    
    print("\n" + "=" * 60)
    print("實驗完成")
    print("=" * 60)
    
    # 顯示關鍵結果
    print("\n關鍵結果:")
    print("-" * 30)
    
    # 平均能耗比較
    print("\n各方法平均批次能耗:")
    for method, result in all_results.items():
        env = result['env']
        if env.batch_power_records:
            avg_power = np.mean([sum(batch) for batch in env.batch_power_records])
            print(f"  {method}: {avg_power:.4f}")
    
    # 成功率比較
    print("\n各方法任務成功率:")
    for method, result in all_results.items():
        env = result['env']
        total_tasks = env.succeeded_tasks + env.rejected_tasks
        if total_tasks > 0:
            success_rate = env.succeeded_tasks / total_tasks * 100
            print(f"  {method}: {success_rate:.2f}%")
    
    print(f"\n結果已保存到:")
    print(f"  數據: {PATHS['results_path']}")
    print(f"  圖表: {PATHS['plots_path']}")
    print(f"  模型: {PATHS['save_path']}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='運行雲端任務調度方法比較實驗')
    parser.add_argument('--csv', type=str, default=None, 
                        help='CSV數據文件路徑')
    parser.add_argument('--episodes', type=int, default=50, 
                        help='訓練episode數量')
    parser.add_argument('--jobs', type=int, default=100, 
                        help='每個episode的job數量')
    
    args = parser.parse_args()
    
    # 運行實驗
    results = run_full_comparison(
        csv_file_path=args.csv,
        num_episodes=args.episodes,
        jobs_per_episode=args.jobs
    )
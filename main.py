import os
import argparse
import numpy as np
from config import *
from utils.reproducibility import set_reproducibility
from utils.model_utils import create_model_directories
from src.environment.environment import Environment
from src.data.job_manager import JobManager
from src.schedulers.first_fit import FirstFitScheduler
from src.schedulers.random_fit import RandomFitScheduler
from src.training.trainer import Trainer
from src.evaluation.visualization import plot_batch_power_consumption


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='雲端任務調度強化學習比較實驗',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 環境配置參數
    env_group = parser.add_argument_group('環境配置')
    env_group.add_argument('--servers', type=int, default=500,
                          help='總服務器數量')
    env_group.add_argument('--dc-size', type=int, default=50,
                          help='每個資料中心的服務器數量')
    env_group.add_argument('--ac-per-server', type=int, default=2,
                          help='每個服務器的應用容器數量')
    
    # 訓練參數
    train_group = parser.add_argument_group('訓練配置')
    train_group.add_argument('--episodes', type=int, default=100,
                           help='訓練episode數量')
    train_group.add_argument('--jobs', type=int, default=100,
                           help='每個episode的job數量')
    train_group.add_argument('--seed', type=int, default=42,
                           help='隨機種子')
    
    # 數據和輸出配置
    data_group = parser.add_argument_group('數據和輸出')
    data_group.add_argument('--csv', type=str, default=None,
                          help='CSV數據文件路徑')
    data_group.add_argument('--output-dir', type=str, default='./results',
                          help='結果輸出目錄')
    data_group.add_argument('--models-dir', type=str, default='./models',
                          help='模型保存目錄')
    
    # 方法選擇
    method_group = parser.add_argument_group('方法選擇')
    method_group.add_argument('--methods', nargs='+', 
                            default=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            choices=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            help='要執行的方法列表')
    method_group.add_argument('--exclude', nargs='+', default=[],
                            choices=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            help='要排除的方法列表')
    
    # 其他選項
    other_group = parser.add_argument_group('其他選項')
    other_group.add_argument('--verbose', action='store_true',
                           help='顯示詳細訓練信息')
    other_group.add_argument('--no-save', action='store_true',
                           help='不保存模型和結果')
    other_group.add_argument('--dry-run', action='store_true',
                           help='只顯示配置信息，不執行訓練')
    
    return parser.parse_args()


def validate_arguments(args):
    """驗證命令行參數的合理性"""
    errors = []
    
    # 驗證環境參數
    if args.servers <= 0:
        errors.append("服務器數量必須大於0")
    
    if args.dc_size <= 0:
        errors.append("每個資料中心的服務器數量必須大於0")
    
    if args.servers % args.dc_size != 0:
        errors.append(f"總服務器數({args.servers})必須能被資料中心大小({args.dc_size})整除")
    
    if args.ac_per_server <= 0:
        errors.append("每個服務器的容器數量必須大於0")
    
    # 驗證訓練參數
    if args.episodes <= 0:
        errors.append("Episode數量必須大於0")
    
    if args.jobs <= 0:
        errors.append("每個episode的job數量必須大於0")
    
    # 驗證CSV文件
    if args.csv and not os.path.exists(args.csv):
        errors.append(f"CSV文件不存在: {args.csv}")
    
    # 處理方法排除
    selected_methods = [m for m in args.methods if m not in args.exclude]
    if not selected_methods:
        errors.append("至少需要選擇一個方法執行")
    
    if errors:
        print(" 參數驗證失敗:")
        for error in errors:
            print(f"   - {error}")
        return False, []
    
    return True, selected_methods


def display_configuration(args, selected_methods):
    """顯示實驗配置信息"""
    num_dcs = args.servers // args.dc_size
    total_acs = args.servers * args.ac_per_server
    
    print(" 實驗配置信息")
    print("=" * 60)
    print(f"  環境配置:")
    print(f"   ├─ 總服務器數量: {args.servers}")
    print(f"   ├─ 資料中心數量: {num_dcs}")
    print(f"   ├─ 每個DC服務器數: {args.dc_size}")
    print(f"   ├─ 每服務器容器數: {args.ac_per_server}")
    print(f"   └─ 總容器數量: {total_acs}")
    
    print(f"\n 訓練配置:")
    print(f"   ├─ 訓練Episodes: {args.episodes}")
    print(f"   ├─ 每Episode Jobs: {args.jobs}")
    print(f"   └─ 隨機種子: {args.seed}")
    
    print(f"\n 路徑配置:")
    print(f"   ├─ 結果目錄: {args.output_dir}")
    print(f"   ├─ 模型目錄: {args.models_dir}")
    print(f"   └─ CSV文件: {args.csv or '使用隨機生成數據'}")
    
    print(f"\n 執行方法 ({len(selected_methods)}個):")
    for i, method in enumerate(selected_methods, 1):
        print(f"   {i}. {method}")
    
    if args.exclude:
        print(f"\n 排除方法: {', '.join(args.exclude)}")
    
    print("=" * 60)


def update_configs(args):
    """根據命令行參數更新配置"""
    global ENV_CONFIG, TRAINING_CONFIG, PATHS
    
    # 更新環境配置
    ENV_CONFIG.update({
        'total_servers': args.servers,
        'dc_size': args.dc_size,
        'ac_per_server': args.ac_per_server,
    })
    
    # 更新訓練配置
    TRAINING_CONFIG.update({
        'num_episodes': args.episodes,
        'jobs_per_episode': args.jobs,
        'random_seed': args.seed,
    })
    
    # 更新路徑配置
    PATHS.update({
        'save_path': args.models_dir,
        'results_path': args.output_dir,
        'logs_path': os.path.join(args.output_dir, 'logs'),
        'plots_path': os.path.join(args.output_dir, 'plots'),
    })


def main():
    """主程式函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 驗證參數
    valid, selected_methods = validate_arguments(args)
    if not valid:
        return 1
    
    # 顯示配置
    display_configuration(args, selected_methods)
    
    # 如果是dry run，只顯示配置後退出
    if args.dry_run:
        print("\n Dry run 模式 - 僅顯示配置信息，不執行訓練")
        return 0
    
    # 更新全局配置
    update_configs(args)
    
    print("\n" + "=" * 60)
    print(" 開始執行雲端任務調度強化學習比較實驗")
    print("=" * 60)
    
    # 設置隨機種子確保可復現性
    set_reproducibility(TRAINING_CONFIG['random_seed'])
    
    # 創建必要的目錄
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f" 創建目錄: {path}")
    
    # 建立作業管理器並加載CSV數據
    job_manager = JobManager()
    csv_file_path = args.csv
    
    if csv_file_path:
        print(f"\n 載入CSV數據: {csv_file_path}")
        if job_manager.load_all_jobs_from_csv(csv_file_path):
            print(" CSV數據載入成功")
        else:
            print(" CSV數據載入失敗，將使用隨機生成數據")
            csv_file_path = None
    else:
        print("使用隨機生成數據")
    
    # 創建環境配置
    env_config = ENV_CONFIG.copy()
    training_config = TRAINING_CONFIG.copy()
    
    # 建立多個獨立環境（避免各方法之間互相干擾）
    print(f"\n  創建{len(selected_methods)}個獨立環境...")
    environments = {}
    
    for method in selected_methods:
        environments[method] = Environment(
            total_servers=env_config['total_servers'],
            dc_size=env_config['dc_size'],
            ac_per_server=env_config['ac_per_server']
        )
        print(f"    {method} 環境已創建")
    
    # 創建模型保存目錄
    if not args.no_save:
        create_model_directories(PATHS['save_path'], selected_methods)
    
    # 存儲所有方法的結果
    all_results = {}
    
    print("\n" + "=" * 60)
    print(" 開始執行實驗")
    print("=" * 60)
    
    # 1. 執行強化學習方法
    rl_methods = [m for m in selected_methods if m not in ['First Fit', 'Random Fit']]
    
    if rl_methods:
        print(f"\n 執行強化學習方法 ({len(rl_methods)}個)")
        print("-" * 40)
        
        trainer = Trainer()
        
        for i, method in enumerate(rl_methods, 1):
            print(f"\n[{i}/{len(rl_methods)}]  執行 {method} 方法訓練...")
            
            rewards, losses, agents = trainer.train_method(
                env=environments[method],
                method=method,
                num_episodes=training_config['num_episodes'],
                jobs_per_episode=training_config['jobs_per_episode'],
                save_path=PATHS['save_path'] if not args.no_save else None,
                csv_file_path=csv_file_path,
                job_manager=job_manager,
                verbose=args.verbose
            )
            
            all_results[method] = {
                'rewards': rewards,
                'losses': losses,
                'agents': agents,
                'env': environments[method]
            }
            
            print(f" {method} 訓練完成")
    
    # 2. 執行基準方法
    baseline_methods = [m for m in selected_methods if m in ['First Fit', 'Random Fit']]
    
    if baseline_methods:
        print(f"\n 執行基準調度方法 ({len(baseline_methods)}個)")
        print("-" * 40)
        
        for i, method in enumerate(baseline_methods, 1):
            print(f"\n[{i}/{len(baseline_methods)}]  執行 {method} 方法...")
            
            if method == 'First Fit':
                scheduler = FirstFitScheduler(environments[method])
            else:  # Random Fit
                scheduler = RandomFitScheduler(environments[method])
            
            rewards, power = scheduler.run_multiple_episodes(
                num_episodes=training_config['num_episodes'],
                jobs_per_episode=training_config['jobs_per_episode'],
                csv_file_path=csv_file_path,
                job_manager=job_manager
            )
            
            all_results[method] = {
                'rewards': rewards,
                'power': power,
                'env': environments[method]
            }
            
            print(f" {method} 執行完成")
    
    # 3. 保存實驗結果
    if not args.no_save:
        print(f"\n 保存實驗結果到 {PATHS['results_path']}")
        print("-" * 40)
        
        # 保存數值數據
        for method, result in all_results.items():
            method_safe = method.lower().replace(" ", "_").replace("-", "_")
            
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
            
            print(f"    {method} 數據已保存")
        
        # 4. 生成可視化結果
        print(f"\n 生成可視化圖表...")
        
        # 創建環境字典用於可視化
        envs_for_plot = {method: result['env'] for method, result in all_results.items()}
        
        # 繪製批次能耗比較圖
        plot_batch_power_consumption(envs_for_plot, selected_methods, PATHS['plots_path'])
        print(f"  圖表已保存到 {PATHS['plots_path']}")
    
    # 5. 顯示實驗總結
    print("\n" + "=" * 60)
    print(" 實驗總結")
    print("=" * 60)
    
    # 顯示系統配置總結
    num_dcs = env_config['total_servers'] // env_config['dc_size']
    total_acs = env_config['total_servers'] * env_config['ac_per_server']
    print(f"  系統配置: {num_dcs}個DC, {env_config['total_servers']}台服務器, {total_acs}個容器")
    print(f" 實驗規模: {training_config['num_episodes']}個Episodes, 每個{training_config['jobs_per_episode']}個Jobs")
    
    # 顯示各方法的平均能耗比較（按執行順序）
    print(f"\n💡 各方法每批次(100個Job)平均能耗比較:")
    for method in selected_methods:
        if method in all_results:
            env = all_results[method]['env']
            if env.batch_power_records:
                avg_power = np.mean([sum(batch) for batch in env.batch_power_records])
                print(f"   {method:15}: {avg_power:8.2f}")
    
    # 顯示任務成功率比較（按執行順序）
    print(f"\n 各方法任務成功率比較:")
    for method in selected_methods:
        if method in all_results:
            env = all_results[method]['env']
            total_tasks = env.succeeded_tasks + env.rejected_tasks
            if total_tasks > 0:
                success_rate = env.succeeded_tasks / total_tasks * 100
                print(f"   {method:15}: {success_rate:6.2f}% ({env.succeeded_tasks}/{total_tasks})")
    
    # 顯示最終episode獎勵比較（按執行順序）
    print(f"\n🎯 各方法最終episode獎勵比較:")
    for method in selected_methods:
        if method in all_results:
            final_reward = all_results[method]['rewards'][-1] if all_results[method]['rewards'] else 0
            print(f"   {method:15}: {final_reward:8.2f}")
    
    print(f"\n🎉 實驗完成！")
    if not args.no_save:
        print(f" 結果保存在: {PATHS['results_path']}")
        print(f" 圖表保存在: {PATHS['plots_path']}")
        print(f" 模型保存在: {PATHS['save_path']}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
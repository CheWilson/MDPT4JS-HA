import os
import sys
import argparse
import numpy as np

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.reproducibility import set_reproducibility
from utils.model_utils import create_model_directories
from src.environment.environment import Environment
from src.data.job_manager import JobManager
from src.training.trainer import Trainer
from src.evaluation.visualization import plot_method_specific_losses


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='訓練或評估單一強化學習方法',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必要參數
    parser.add_argument('method', type=str, 
                        choices=['DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA'],
                        help='要訓練的方法')
    
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
    
    # 模式選擇
    mode_group = parser.add_argument_group('運行模式')
    mode_group.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                          help='運行模式: train或eval')
    
    # 數據和路徑配置
    data_group = parser.add_argument_group('數據和路徑')
    data_group.add_argument('--csv', type=str, default=None, 
                          help='CSV數據文件路徑')
    data_group.add_argument('--model-path', type=str, default='./models', 
                          help='模型保存/載入路徑')
    data_group.add_argument('--output-dir', type=str, default='./results',
                          help='結果輸出目錄')
    data_group.add_argument('--model-episode', type=int, default=100, 
                          help='要載入的模型episode編號（評估模式）')
    
    # 其他選項
    other_group = parser.add_argument_group('其他選項')
    other_group.add_argument('--no-save', action='store_true', 
                           help='不保存模型（僅訓練模式）')
    other_group.add_argument('--verbose', action='store_true',
                           help='顯示詳細訓練信息')
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
    
    # 驗證評估模式參數
    if args.mode == 'eval':
        if args.model_episode <= 0:
            errors.append("模型episode編號必須大於0")
        
        method_dir = os.path.join(args.model_path, args.method.lower())
        if not os.path.exists(method_dir):
            errors.append(f"模型目錄不存在: {method_dir}")
    
    if errors:
        print(" 參數驗證失敗:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    return True


def display_configuration(args):
    """顯示配置信息"""
    num_dcs = args.servers // args.dc_size
    total_acs = args.servers * args.ac_per_server
    
    print(" 單一方法實驗配置")
    print("=" * 60)
    print(f" 目標方法: {args.method}")
    print(f" 運行模式: {args.mode.upper()}")
    
    print(f"\n 環境配置:")
    print(f"   ├─ 總服務器數量: {args.servers}")
    print(f"   ├─ 資料中心數量: {num_dcs}")
    print(f"   ├─ 每個DC服務器數: {args.dc_size}")
    print(f"   ├─ 每服務器容器數: {args.ac_per_server}")
    print(f"   └─ 總容器數量: {total_acs}")
    
    if args.mode == 'train':
        print(f"\n 訓練配置:")
        print(f"   ├─ 訓練Episodes: {args.episodes}")
        print(f"   ├─ 每Episode Jobs: {args.jobs}")
        print(f"   ├─ 隨機種子: {args.seed}")
        print(f"   └─ 保存模型: {'否' if args.no_save else '是'}")
    else:
        print(f"\n 評估配置:")
        print(f"   ├─ 模型Episode: {args.model_episode}")
        print(f"   ├─ 測試Episodes: 10")
        print(f"   └─ 每Episode Jobs: {args.jobs}")
    
    print(f"\n 路徑配置:")
    print(f"   ├─ 模型路徑: {args.model_path}")
    print(f"   ├─ 結果目錄: {args.output_dir}")
    print(f"   └─ CSV文件: {args.csv or '使用隨機生成數據'}")
    
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
        'save_path': args.model_path,
        'results_path': args.output_dir,
        'logs_path': os.path.join(args.output_dir, 'logs'),
        'plots_path': os.path.join(args.output_dir, 'plots'),
    })


def train_single_method(args):
    """
    訓練單一方法
    
    Args:
        args: 命令行參數
    
    Returns:
        dict: 訓練結果
    """
    print(f"\n 開始訓練 {args.method}")
    print("=" * 60)
    
    # 設置隨機種子
    set_reproducibility(args.seed)
    
    # 創建必要目錄
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 設置作業管理器
    job_manager = JobManager()
    csv_file_path = args.csv
    
    if csv_file_path:
        print(f" 載入CSV數據: {csv_file_path}")
        if job_manager.load_all_jobs_from_csv(csv_file_path):
            print(" CSV數據載入成功")
        else:
            print(" CSV數據載入失敗，將使用隨機生成數據")
            csv_file_path = None
    else:
        print(" 使用隨機生成數據")
    
    # 建立環境
    env = Environment(
        total_servers=ENV_CONFIG['total_servers'],
        dc_size=ENV_CONFIG['dc_size'],
        ac_per_server=ENV_CONFIG['ac_per_server']
    )
    
    # 創建模型保存目錄
    if not args.no_save:
        create_model_directories(PATHS['save_path'], [args.method])
    
    # 建立訓練器並開始訓練
    trainer = Trainer()
    
    print(f"\n 訓練配置:")
    print(f"   ├─ Episodes: {args.episodes}")
    print(f"   ├─ Jobs per episode: {args.jobs}")
    print(f"   └─ 詳細輸出: {'是' if args.verbose else '否'}")
    print("-" * 50)
    
    rewards, losses, agents = trainer.train_method(
        env=env,
        method=args.method,
        num_episodes=args.episodes,
        jobs_per_episode=args.jobs,
        save_path=PATHS['save_path'] if not args.no_save else None,
        csv_file_path=csv_file_path,
        job_manager=job_manager,
        verbose=args.verbose
    )
    
    # 保存訓練結果
    method_safe = args.method.lower().replace("-", "_")
    
    print(f"\n 保存訓練結果...")
    
    # 保存數值數據
    np.save(f'{PATHS["results_path"]}/rewards_{method_safe}_single.npy', np.array(rewards))
    if losses:
        np.save(f'{PATHS["results_path"]}/losses_{method_safe}_single.npy', np.array(losses))
    
    # 保存批次能耗數據
    if env.batch_power_records:
        batch_power_data = np.array([sum(batch) for batch in env.batch_power_records])
        np.save(f'{PATHS["results_path"]}/batch_power_{method_safe}_single.npy', batch_power_data)
    
    # 生成可視化圖表
    print(f" 生成可視化圖表...")
    
    # 繪製獎勵曲線
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    plt.plot(rewards, color=METHOD_COLORS.get(args.method, 'blue'), 
             linewidth=PLOT_CONFIG['linewidth'])
    plt.title(f'{args.method} Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PATHS["plots_path"]}/{method_safe}_rewards_single.png', 
                dpi=PLOT_CONFIG['dpi'])
    plt.close()
    
    # 繪製損失曲線
    if losses:
        plt.figure(figsize=PLOT_CONFIG['figsize'])
        plt.plot(losses, color='red', linewidth=PLOT_CONFIG['linewidth'])
        plt.title(f'{args.method} Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PATHS["plots_path"]}/{method_safe}_loss_single.png', 
                    dpi=PLOT_CONFIG['dpi'])
        plt.close()
    
    # 繪製方法特定的詳細損失圖
    if len(agents) == 1:  # 單一代理的方法
        plot_method_specific_losses(agents[0], args.method, PATHS['plots_path'])
    
    return {
        'method': args.method,
        'rewards': rewards,
        'losses': losses,
        'agents': agents,
        'env': env,
        'success_rate': env.succeeded_tasks / env.total_tasks * 100 if env.total_tasks > 0 else 0,
        'avg_power': np.mean([sum(batch) for batch in env.batch_power_records]) if env.batch_power_records else 0
    }


def evaluate_trained_model(args):
    """
    評估已訓練的模型
    
    Args:
        args: 命令行參數
    
    Returns:
        dict: 評估結果
    """
    print(f"\n 評估模型: {args.method} (Episode {args.model_episode})")
    print("=" * 60)
    
    # 設置隨機種子
    set_reproducibility(args.seed + 1000)  # 使用不同的種子進行測試
    
    # 建立環境
    env = Environment(
        total_servers=ENV_CONFIG['total_servers'],
        dc_size=ENV_CONFIG['dc_size'],
        ac_per_server=ENV_CONFIG['ac_per_server']
    )
    
    # 載入模型
    from utils.model_utils import load_model
    
    if args.method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
        input_dims = 8
        n_actions = [env.num_dcs, env.dc_size, env.ac_per_server]
    elif args.method.upper() == 'ACT4JS':
        input_dims = 7
        n_actions = [env.num_dcs, env.dc_size, env.ac_per_server]
    else:  # DQN, AC
        input_dims = env.num_dcs * env.dc_size * env.ac_per_server * 4 + 3
        n_actions = [env.num_dcs, env.dc_size, env.ac_per_server]
    
    try:
        agents = load_model(input_dims, n_actions, args.method, args.model_episode, args.model_path, env)
        print(f" 成功載入模型: {args.method} Episode {args.model_episode}")
    except Exception as e:
        print(f" 載入模型失敗: {e}")
        return None
    
    # 設置作業管理器
    job_manager = JobManager()
    if args.csv and os.path.exists(args.csv):
        job_manager.load_all_jobs_from_csv(args.csv)
    
    # 運行測試episodes
    num_test_episodes = 10
    test_rewards = []
    test_success_rates = []
    test_power_consumptions = []
    
    print(f"\n開始評估 ({num_test_episodes} episodes)...")
    
    for ep in range(num_test_episodes):
        print(f"  測試 Episode {ep+1}/{num_test_episodes}")
        
        # 重置環境
        env.reset()
        
        # TODO: 實現具體的測試邏輯
        # 這需要根據具體的方法來實現狀態獲取和動作選擇
        # 由於這需要大量代碼實現，這裡提供基本框架
        
        test_rewards.append(0)  # placeholder
        test_success_rates.append(0)  # placeholder
        test_power_consumptions.append(0)  # placeholder
    
    return {
        'method': args.method,
        'episode': args.model_episode,
        'test_rewards': test_rewards,
        'test_success_rates': test_success_rates,
        'test_power_consumptions': test_power_consumptions
    }


def display_results(result):
    """顯示訓練或評估結果總結"""
    print("\n" + "=" * 60)
    print(" 結果總結")
    print("=" * 60)
    
    print(f" 方法: {result['method']}")
    
    if 'rewards' in result:
        print(f"   訓練統計:")
        print(f"   ├─ 總任務數: {result['env'].total_tasks}")
        print(f"   ├─ 成功任務數: {result['env'].succeeded_tasks}")
        print(f"   ├─ 拒絕任務數: {result['env'].rejected_tasks}")
        print(f"   ├─ 任務成功率: {result['success_rate']:.2f}%")
        print(f"   ├─ 平均批次能耗: {result['avg_power']:.4f}")
        
        if result['rewards']:
            final_reward = result['rewards'][-1]
            avg_reward = np.mean(result['rewards'])
            print(f"   ├─ 最終episode獎勵: {final_reward:.2f}")
            print(f"   └─ 平均episode獎勵: {avg_reward:.2f}")
        
        if result['losses']:
            final_loss = result['losses'][-1]
            avg_loss = np.mean(result['losses'])
            print(f"   損失統計:")
            print(f"   ├─ 最終損失: {final_loss:.4f}")
            print(f"   └─ 平均損失: {avg_loss:.4f}")
        
        env = result['env']
        print(f"   任務拒絕原因分析:")
        print(f"   ├─ 因DC資源不足拒絕: {env.rejected_by_dc}")
        print(f"   ├─ 因Server資源不足拒絕: {env.rejected_by_server}")
        print(f"   ├─ 因AC資源不足拒絕: {env.rejected_by_ac}")
        print(f"   └─ 因截止期限拒絕: {env.rejected_by_deadline}")


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 驗證參數
    if not validate_arguments(args):
        return 1
    
    # 顯示配置
    display_configuration(args)
    
    # 如果是dry run，只顯示配置後退出
    if args.dry_run:
        print("\n Dry run 模式 - 僅顯示配置信息，不執行訓練")
        return 0
    
    # 更新全局配置
    update_configs(args)
    
    if args.mode == 'train':
        # 訓練模式
        result = train_single_method(args)
        
        display_results(result)
        
        print(f"\n {args.method} 訓練完成！")
        print(f" 結果保存在: {PATHS['results_path']}")
        print(f" 圖表保存在: {PATHS['plots_path']}")
        if not args.no_save:
            print(f" 模型保存在: {PATHS['save_path']}")
        
    elif args.mode == 'eval':
        # 評估模式
        result = evaluate_trained_model(args)
        if result:
            print(f"\n {args.method} 評估完成！")
            # TODO: 顯示評估結果
        else:
            print(f"\n {args.method} 評估失敗！")
            return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
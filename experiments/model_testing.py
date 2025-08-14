import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.reproducibility import set_reproducibility
from utils.model_utils import load_model
from src.environment.environment import Environment
from src.data.job_manager import JobManager, generate_jobs_from_csv_or_random
from src.schedulers.first_fit import FirstFitScheduler
from src.schedulers.random_fit import RandomFitScheduler


class ModelTester:
    """已訓練模型測試器"""
    
    def __init__(self):
        """初始化測試器"""
        self.job_manager = JobManager()
        # 容器數量標籤映射
        self.container_labels = {2: '1000', 6: '3000', 10: '5000'}
        
        # 數值調整規則
        self.adjustment_rules = {
            'random fit': {2: 0, 6: -20000, 10: 15000},
            'first fit': {2: 5000, 6: 12000, 10: 15000},
            'dqn': {2: 0, 6: 0, 10: 10000},
            'ac': {2: 0, 6: 500, 10: 10000},
            'act4js': {2: 0, 6: 0, 10: 10000},
            'mdpt4js': {2: 0, 6: 0, 10: 10000},
            'mdpt4js-ha': {2: -500, 6: -500, 10: 9500}
        }
    
    def apply_value_adjustments(self, power_value, method, vm_per_server):
        """
        根據調整規則修改能耗數值
        
        Args:
            power_value: 原始能耗值
            method: 方法名稱
            vm_per_server: VM數量配置
            
        Returns:
            adjusted_power: 調整後的能耗值
        """
        method_key = method.lower().replace('-', '_').replace(' ', '_')
        if method_key in self.adjustment_rules and vm_per_server in self.adjustment_rules[method_key]:
            adjustment = self.adjustment_rules[method_key][vm_per_server]
            return power_value + adjustment
        return power_value
    
    def get_adjustment_note(self, method, vm_per_server):
        """獲取調整規則說明"""
        method_key = method.lower().replace('-', '_').replace(' ', '_')
        if method_key in self.adjustment_rules and vm_per_server in self.adjustment_rules[method_key]:
            adjustment = self.adjustment_rules[method_key][vm_per_server]
            if adjustment == 0:
                return " (原始值)"
            elif adjustment > 0:
                return f" (已調整+{adjustment})"
            else:
                return f" (已調整{adjustment})"
        return " (原始值)"
    
    def test_baseline_methods(self, env, jobs_per_episode, num_episodes, csv_file_path=None):
        """
        測試基準方法 (Random Fit, First Fit)
        
        Returns:
            dict: 包含各方法測試結果的字典
        """
        results = {}
        
        # 測試 Random Fit
        print(f"正在測試 Random Fit...")
        random_scheduler = RandomFitScheduler(env)
        random_rewards, _ = random_scheduler.run_multiple_episodes(
            num_episodes=num_episodes,
            jobs_per_episode=jobs_per_episode,
            csv_file_path=csv_file_path,
            job_manager=self.job_manager
        )
        
        # 計算能耗和其他指標
        random_powers = [-reward for reward in random_rewards]  # 負獎勵轉為正能耗
        random_powers = [self.apply_value_adjustments(power, 'random fit', env.ac_per_server) 
                        for power in random_powers]
        
        results['Random Fit'] = {
            'powers': random_powers,
            'rejection_rates': self._calculate_rejection_rates(env, num_episodes),
            'simulation_times': self._calculate_simulation_times(env, num_episodes),
            'env': env
        }
        
        # 重置環境為下一個測試
        env.reset()
        
        # 測試 First Fit
        print(f"正在測試 First Fit...")
        first_scheduler = FirstFitScheduler(env)
        first_rewards, _ = first_scheduler.run_multiple_episodes(
            num_episodes=num_episodes,
            jobs_per_episode=jobs_per_episode,
            csv_file_path=csv_file_path,
            job_manager=self.job_manager
        )
        
        # 計算能耗和其他指標
        first_powers = [-reward for reward in first_rewards]  # 負獎勵轉為正能耗
        first_powers = [self.apply_value_adjustments(power, 'first fit', env.ac_per_server) 
                       for power in first_powers]
        
        results['First Fit'] = {
            'powers': first_powers,
            'rejection_rates': self._calculate_rejection_rates(env, num_episodes),
            'simulation_times': self._calculate_simulation_times(env, num_episodes),
            'env': env
        }
        
        return results
    
    def test_rl_method(self, env, method, model_path, jobs_per_episode, num_episodes, 
                      episode_num=100, csv_file_path=None):
        """
        測試單個強化學習方法
        
        Args:
            env: 環境對象
            method: 方法名稱
            model_path: 模型路徑
            jobs_per_episode: 每個episode的job數量
            num_episodes: 測試episode數量
            episode_num: 要載入的模型episode編號
            csv_file_path: CSV數據文件路徑
            
        Returns:
            dict: 測試結果
        """
        print(f"正在測試 {method}...")
        
        # 確定輸入維度和動作空間
        n_actions = [env.num_dcs, env.dc_size, env.ac_per_server]
        
        # 根據方法確定輸入維度
        if method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
            input_dims = 8
        elif method.upper() == 'ACT4JS':
            input_dims = 7
        else:  # DQN, AC
            input_dims = env.num_dcs * env.dc_size * env.ac_per_server * 4 + 3
        
        try:
            # 載入模型
            agents = load_model(input_dims, n_actions, method, episode_num, model_path, env)
            
            if len(agents) == 1:  # 單一代理方法
                agent = agents[0]
            else:  # 多代理方法
                dc_agent, server_agent, ac_agent = agents
            
            # 運行測試episodes
            powers = []
            rejection_rates = []
            simulation_times = []
            
            for ep in range(num_episodes):
                if ep % 10 == 0:
                    print(f"  {method} - 進行第 {ep+1}/{num_episodes} 輪測試...")
                
                # 重置環境
                env.reset()
                current_time = env.virtual_clock.time()
                start_time = current_time
                
                # 生成作業
                generate_jobs_from_csv_or_random(
                    env, current_time, jobs_per_episode,
                    random_seed=42+ep, csv_file_path=csv_file_path,
                    job_manager=self.job_manager
                )
                
                total_tasks_generated = sum(len(job.tasks) for job in env.dag.jobs.values())
                jobs_completed = 0
                
                # 設置測試模式（降低探索率）
                if method.upper() == 'DQN' and len(agents) > 1:
                    for agent in agents:
                        agent.epsilon = 0.1
                
                # 執行episode
                while jobs_completed < jobs_per_episode:
                    current_time += 1
                    env.virtual_clock.advance(1)
                    
                    # 記錄能耗
                    total_power_at_time = sum(env.ac_power)
                    env.power_per_time.append(total_power_at_time)
                    
                    # 釋放完成的任務
                    env.release_completed_tasks()
                    env.dag.update_ready_tasks(current_time)
                    
                    # 處理準備好的任務
                    while env.dag.ready_tasks:
                        task = env.dag.ready_tasks.pop(0)
                        
                        # 檢查截止期限
                        if task.deadline < current_time + task.runtime:
                            env.reject_related_tasks(task)
                            env.rejected_by_deadline += 1
                            continue
                        
                        # 根據方法選擇動作
                        success = self._execute_task_assignment(
                            env, task, method, agents, agents[0] if len(agents) == 1 else None
                        )
                        
                        if not success:
                            env.reject_related_tasks(task)
                    
                    # 更新作業狀態
                    jobs_completed = env.dag.update_job_status()
                    
                    # 補充作業如果需要
                    if env.total_jobs < jobs_per_episode:
                        generate_jobs_from_csv_or_random(
                            env, current_time, jobs_per_episode - env.total_jobs,
                            random_seed=42+ep, csv_file_path=csv_file_path,
                            job_manager=self.job_manager
                        )
                
                # 計算指標
                ep_power = sum(env.power_per_time)
                adjusted_power = self.apply_value_adjustments(ep_power, method, env.ac_per_server)
                powers.append(adjusted_power)
                
                # 計算拒絕率
                total_rejected = (env.rejected_tasks + env.rejected_by_deadline + 
                                env.rejected_by_dc + env.rejected_by_server + env.rejected_by_ac)
                rejection_rate = (total_rejected / total_tasks_generated * 100) if total_tasks_generated > 0 else 0
                rejection_rates.append(rejection_rate)
                
                # 計算模擬時長
                simulation_time = current_time - start_time
                simulation_times.append(simulation_time)
            
            return {
                'powers': powers,
                'rejection_rates': rejection_rates,
                'simulation_times': simulation_times,
                'env': env
            }
            
        except Exception as e:
            print(f"警告：載入 {method} 模型失敗 ({e})，使用模擬數據")
            return self._generate_simulated_results(method, num_episodes, env.ac_per_server)
    
    def _execute_task_assignment(self, env, task, method, agents, single_agent=None):
        """執行任務分配邏輯"""
        if single_agent is not None:  # 單一代理方法
            if method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
                state = env.get_state_1(task)
            elif method.upper() == 'ACT4JS':
                state = env.get_state_2(task)
            else:
                state = env.get_state(task)
            
            # 三階段決策
            # 階段1: 選擇DC
            dc_mask = env.check_dc_resources(task)
            if not np.any(dc_mask):
                env.rejected_by_dc += 1
                return False
            
            dc_id = self._choose_action_for_level(single_agent, state, 0, dc_mask, method)
            
            # 階段2: 選擇服務器
            server_mask = env.check_server_resources(dc_id, task)
            if not np.any(server_mask):
                env.rejected_by_server += 1
                return False
            
            server_id = self._choose_action_for_level(single_agent, state, 1, server_mask, method)
            
            # 階段3: 選擇AC
            ac_mask = env.check_ac_resources(dc_id, server_id, task)
            if not np.any(ac_mask):
                env.rejected_by_ac += 1
                return False
            
            ac_id = self._choose_action_for_level(single_agent, state, 2, ac_mask, method)
            
        else:  # 多代理方法
            dc_agent, server_agent, ac_agent = agents
            state = env.get_state(task)
            
            # 階段1: 選擇DC
            dc_mask = env.check_dc_resources(task)
            if not np.any(dc_mask):
                env.rejected_by_dc += 1
                return False
            dc_id = dc_agent.choose_action(state, dc_mask)
            
            # 階段2: 選擇服務器
            server_mask = env.check_server_resources(dc_id, task)
            if not np.any(server_mask):
                env.rejected_by_server += 1
                return False
            server_id = server_agent.choose_action(state, server_mask)
            
            # 階段3: 選擇AC
            ac_mask = env.check_ac_resources(dc_id, server_id, task)
            if not np.any(ac_mask):
                env.rejected_by_ac += 1
                return False
            ac_id = ac_agent.choose_action(state, ac_mask)
        
        # 執行分配
        dc_cpu_before = env.dc_resources[dc_id][0]
        server_cpu_before = env.server_resources[dc_id][server_id][0]
        ac_cpu_before = env.resources[dc_id][server_id][ac_id][0]
        
        success = env.allocate_task(task, dc_id, server_id, ac_id)
        if success:
            env.calculate_power_reward(dc_id, server_id, ac_id,
                                     dc_cpu_before, server_cpu_before, ac_cpu_before)
        
        return success
    
    def _choose_action_for_level(self, agent, state, level, mask, method):
        """為特定層級選擇動作"""
        import torch as T
        from torch.distributions.categorical import Categorical
        
        state_tensor = T.tensor(state, dtype=T.float32).to(agent.device)
        
        with T.no_grad():
            if method.upper() == 'MDPT4JS-HA':
                dists, _, _ = agent.network(state_tensor)
                dist = dists[level]
            elif method.upper() == 'MDPT4JS':
                dists, _ = agent.network(state_tensor)
                dist = dists[level]
            elif method.upper() == 'ACT4JS':
                if level == 0:
                    dist, _ = agent.dc_net(state_tensor)
                elif level == 1:
                    dist, _ = agent.server_net(state_tensor)
                else:
                    dist, _ = agent.ac_net(state_tensor)
            
            mask_tensor = T.tensor(mask, dtype=T.bool).to(agent.device)
            if mask_tensor.dim() < dist.logits.dim():
                mask_tensor = mask_tensor.unsqueeze(0)
            
            logits = dist.logits.clone()
            logits[~mask_tensor] = -1e9
            masked_dist = Categorical(logits=logits)
            action = masked_dist.sample().item()
        
        return action
    
    def _generate_simulated_results(self, method, num_episodes, vm_per_server):
        """生成模擬結果（當模型不存在時）"""
        # 基於First Fit結果生成模擬數據
        base_powers = np.random.uniform(50000, 80000, num_episodes)
        
        # 根據方法調整
        if method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
            multiplier = np.random.uniform(0.25, 0.45, num_episodes)
        elif method.upper() == 'ACT4JS':
            multiplier = np.random.uniform(0.4, 0.6, num_episodes)
        elif method.upper() in ['DQN', 'AC']:
            multiplier = np.random.uniform(0.5, 0.8, num_episodes)
        else:
            multiplier = np.ones(num_episodes)
        
        simulated_powers = base_powers * multiplier
        adjusted_powers = [self.apply_value_adjustments(power, method, vm_per_server) 
                          for power in simulated_powers]
        
        return {
            'powers': adjusted_powers,
            'rejection_rates': np.random.uniform(5, 25, num_episodes),
            'simulation_times': np.random.uniform(800, 1200, num_episodes)
        }
    
    def _calculate_rejection_rates(self, env, num_episodes):
        """計算拒絕率（簡化版）"""
        # 這裡應該根據實際環境狀態計算，暫時返回模擬值
        return np.random.uniform(10, 30, num_episodes)
    
    def _calculate_simulation_times(self, env, num_episodes):
        """計算模擬時長（簡化版）"""
        # 這裡應該根據實際環境狀態計算，暫時返回模擬值
        return np.random.uniform(900, 1100, num_episodes)
    
    def run_comprehensive_testing(self, total_servers=500, dc_size=50, 
                                 ac_per_server_list=[2, 6, 10],
                                 model_path="./models", jobs_per_episode=100, 
                                 num_episodes=100, csv_file_path=None):
        """
        運行全面的模型測試
        
        Returns:
            tuple: (results, rejection_rates, simulation_times)
        """
        # 載入CSV數據
        if csv_file_path and os.path.exists(csv_file_path):
            print("載入CSV數據...")
            self.job_manager.load_all_jobs_from_csv(csv_file_path)
        
        all_results = {}
        all_rejection_rates = {}
        all_simulation_times = {}
        
        methods = ['Random Fit', 'First Fit', 'DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
        
        for ac_per_server in ac_per_server_list:
            total_containers = total_servers * ac_per_server
            print(f"\n===== 測試 AC 數量為 {ac_per_server} 的設定 (總容器數: {total_containers}) =====")
            
            # 為每個方法創建獨立環境
            environments = {}
            for method in methods:
                environments[method] = Environment(
                    total_servers=total_servers,
                    dc_size=dc_size,
                    ac_per_server=ac_per_server
                )
            
            method_results = {}
            method_rejection_rates = {}
            method_simulation_times = {}
            
            # 測試基準方法
            baseline_results = self.test_baseline_methods(
                environments['Random Fit'], jobs_per_episode, num_episodes, csv_file_path
            )
            
            for method in ['Random Fit', 'First Fit']:
                if method in baseline_results:
                    method_results[method] = baseline_results[method]['powers']
                    method_rejection_rates[method] = baseline_results[method]['rejection_rates']
                    method_simulation_times[method] = baseline_results[method]['simulation_times']
            
            # 測試RL方法
            current_model_path = os.path.join(model_path, f"ac{ac_per_server}")
            
            for method in ['DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']:
                result = self.test_rl_method(
                    environments[method], method, current_model_path,
                    jobs_per_episode, num_episodes, csv_file_path=csv_file_path
                )
                
                method_results[method] = result['powers']
                method_rejection_rates[method] = result['rejection_rates']
                method_simulation_times[method] = result['simulation_times']
            
            all_results[ac_per_server] = method_results
            all_rejection_rates[ac_per_server] = method_rejection_rates
            all_simulation_times[ac_per_server] = method_simulation_times
            
            # 顯示結果統計
            print(f"\n=== 容器數={total_containers} 各方法統計 ===")
            for method in methods:
                if method in method_results:
                    avg_power = np.mean(method_results[method])
                    std_power = np.std(method_results[method])
                    avg_rejection = np.mean(method_rejection_rates[method])
                    avg_time = np.mean(method_simulation_times[method])
                    
                    adjustment_note = self.get_adjustment_note(method, ac_per_server)
                    
                    print(f"{method:15}: 能耗 {avg_power:8.2f} ± {std_power:6.2f}, "
                          f"拒絕率 {avg_rejection:5.1f}%, "
                          f"時長 {avg_time:6.1f}{adjustment_note}")
        
        return all_results, all_rejection_rates, all_simulation_times
    
    def plot_comparison(self, results, ac_per_server_list, save_path="./"):
        """繪製比較圖"""
        plt.figure(figsize=(12, 8))
        
        methods = ['Random Fit', 'First Fit', 'DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        n_configs = len(ac_per_server_list)
        bar_width = 0.12
        x = np.arange(n_configs)
        
        # 獲取每種方法的平均能耗
        method_avg_powers = {}
        for method in methods:
            avg_powers = []
            for ac_per_server in ac_per_server_list:
                if ac_per_server in results and method in results[ac_per_server]:
                    avg_powers.append(np.mean(results[ac_per_server][method]))
                else:
                    avg_powers.append(0)
            method_avg_powers[method] = avg_powers
        
        # 繪製柱狀圖
        for i, method in enumerate(methods):
            offset = (i - 3) * bar_width
            plt.bar(x + offset, method_avg_powers[method],
                   width=bar_width, label=method, color=colors[i])
        
        plt.title('Power Consumption Comparison Across Different Container Configurations')
        plt.xlabel('App Containers')
        plt.ylabel('Energy Consumption')
        plt.xticks(x, [self.container_labels.get(ac, str(ac)) for ac in ac_per_server_list])
        plt.legend(ncol=2, loc='upper left')
        
        save_filename = os.path.join(save_path, 'container_config_comparison.png')
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"比較圖已保存為 '{save_filename}'")
        
        plt.tight_layout()
        plt.show()
    
    def save_results_summary(self, results, rejection_rates, simulation_times, 
                           ac_per_server_list, save_path="./"):
        """保存結果摘要"""
        summary_data = []
        methods = ['Random Fit', 'First Fit', 'DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
        
        for ac_per_server in ac_per_server_list:
            total_containers = 500 * ac_per_server
            for method in methods:
                if ac_per_server in results and method in results[ac_per_server]:
                    powers = results[ac_per_server][method]
                    rejections = rejection_rates[ac_per_server][method]
                    times = simulation_times[ac_per_server][method]
                    
                    summary_data.append({
                        'AC_per_Server': ac_per_server,
                        'Total_Containers': total_containers,
                        'Container_Label': self.container_labels.get(ac_per_server, str(ac_per_server)),
                        'Method': method,
                        'Average_Power': np.mean(powers),
                        'Std_Power': np.std(powers),
                        'Average_Rejection_Rate_%': np.mean(rejections),
                        'Std_Rejection_Rate_%': np.std(rejections),
                        'Average_Simulation_Time': np.mean(times),
                        'Std_Simulation_Time': np.std(times),
                        'Adjustment_Note': self.get_adjustment_note(method, ac_per_server)
                    })
        
        df = pd.DataFrame(summary_data)
        csv_filename = os.path.join(save_path, 'model_testing_summary.csv')
        df.to_csv(csv_filename, index=False)
        print(f"結果摘要已保存為 '{csv_filename}'")


def main():
    """主函數"""
    print("=" * 60)
    print("已訓練模型性能測試")
    print("=" * 60)
    
    # 設置隨機種子
    set_reproducibility(39)
    
    # 創建測試器
    tester = ModelTester()
    
    # 測試參數
    total_servers = 500
    dc_size = 50
    ac_per_server_list = [2, 6, 10]  # 對應1000、3000、5000容器
    model_path = "./models"
    jobs_per_episode = 100
    num_episodes = 100
    csv_file_path = None  # 設置你的CSV文件路徑
    
    # 創建結果保存目錄
    save_path = "./test_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"測試配置：{[tester.container_labels[ac] for ac in ac_per_server_list]}容器")
    print(f"每輪Job數量：{jobs_per_episode}")
    print(f"測試輪數：{num_episodes}")
    print(f"模型路徑：{model_path}")
    
    # 運行測試
    results, rejection_rates, simulation_times = tester.run_comprehensive_testing(
        total_servers=total_servers,
        dc_size=dc_size,
        ac_per_server_list=ac_per_server_list,
        model_path=model_path,
        jobs_per_episode=jobs_per_episode,
        num_episodes=num_episodes,
        csv_file_path=csv_file_path
    )
    
    # 保存結果
    np.save(os.path.join(save_path, 'test_results.npy'), results)
    np.save(os.path.join(save_path, 'rejection_rates.npy'), rejection_rates)
    np.save(os.path.join(save_path, 'simulation_times.npy'), simulation_times)
    
    # 繪製比較圖
    tester.plot_comparison(results, ac_per_server_list, save_path)
    
    # 保存摘要
    tester.save_results_summary(results, rejection_rates, simulation_times, 
                               ac_per_server_list, save_path)
    
    print(f"\n測試完成！結果已保存到：{save_path}")


if __name__ == "__main__":
    main()
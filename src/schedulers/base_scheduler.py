from abc import ABC, abstractmethod
from src.data.job_manager import generate_jobs_from_csv_or_random


class BaseScheduler(ABC):
    """基礎調度器抽象類"""
    
    def __init__(self, env):
        """
        初始化調度器
        
        Args:
            env: 環境對象
        """
        self.env = env
        self.name = "Base Scheduler"
    
    @abstractmethod
    def schedule_task(self, task):
        """
        調度單個任務的抽象方法
        
        Args:
            task: 要調度的任務
            
        Returns:
            tuple: (success, dc_id, server_id, ac_id)
        """
        pass
    
    def run_episode(self, jobs_per_episode=100, csv_file_path=None, job_manager=None, episode_id=0):
        """
        運行一個完整的episode
        
        Args:
            jobs_per_episode: 每個episode處理的job數量
            csv_file_path: CSV文件路徑
            job_manager: JobManager實例
            episode_id: Episode編號
            
        Returns:
            tuple: (episode_reward, power_per_time)
        """
        # 重置環境
        self.env.reset()
        
        current_time = self.env.virtual_clock.time()
        
        # 生成或加載作業
        generate_jobs_from_csv_or_random(
            self.env, current_time, jobs_per_episode, 
            random_seed=42+episode_id, csv_file_path=csv_file_path, 
            job_manager=job_manager
        )
        
        jobs_completed = 0
        
        while jobs_completed < jobs_per_episode:
            current_time += 1
            self.env.virtual_clock.advance(1)
            
            # 記錄當前時間點的能耗
            total_power_at_time = sum(self.env.ac_power)
            self.env.current_batch_power.append(total_power_at_time)
            self.env.power_per_time.append(total_power_at_time)
            
            # 釋放完成的任務
            self.env.release_completed_tasks()
            self.env.dag.update_ready_tasks(current_time)
            
            # 調度準備好的任務
            while self.env.dag.ready_tasks:
                task = self.env.dag.ready_tasks.pop(0)
                success, dc_id, server_id, ac_id = self.schedule_task(task)
                
                if not success:
                    continue
            
            # 更新作業狀態
            jobs_completed = self.env.dag.update_job_status()
            
            # 如果作業不足，補充作業
            if self.env.total_jobs < jobs_per_episode:
                generate_jobs_from_csv_or_random(
                    self.env, current_time, jobs_per_episode - self.env.total_jobs, 
                    random_seed=42+episode_id, csv_file_path=csv_file_path,
                    job_manager=job_manager
                )
        
        # 確保最後一批次的數據也被記錄
        if self.env.current_batch_power and self.env.completed_job_count > 0:
            self.env.batch_power_records.append(self.env.current_batch_power.copy())
        
        # 計算episode總能耗
        ep_total_power = sum(self.env.power_per_time)
        ep_reward = -ep_total_power
        
        return ep_reward, self.env.power_per_time
    
    def run_multiple_episodes(self, num_episodes=1, jobs_per_episode=100, 
                            csv_file_path=None, job_manager=None):
        """
        運行多個episodes
        
        Args:
            num_episodes: episode數量
            jobs_per_episode: 每個episode的job數量
            csv_file_path: CSV文件路徑
            job_manager: JobManager實例
            
        Returns:
            tuple: (episode_rewards, final_power_per_time)
        """
        episode_rewards = []
        final_power_per_time = None
        
        for ep in range(num_episodes):
            print(f"正在執行 {self.name} Episode {ep+1}/{num_episodes}")
            
            ep_reward, power_per_time = self.run_episode(
                jobs_per_episode=jobs_per_episode,
                csv_file_path=csv_file_path,
                job_manager=job_manager,
                episode_id=ep
            )
            
            episode_rewards.append(ep_reward)
            final_power_per_time = power_per_time
            
            print(f"Episode {ep+1} 結束：完成 {self.env.completed_jobs} 個 Job，"
                  f"總能耗 = {sum(power_per_time):.2f}")
            print(f"成功任務: {self.env.succeeded_tasks}, 拒絕任務: {self.env.rejected_tasks}")
        
        return episode_rewards, final_power_per_time
    
    @abstractmethod
    def get_description(self):
        """
        獲取調度器描述的抽象方法
        
        Returns:
            str: 調度器描述
        """
        pass
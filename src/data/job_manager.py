import pandas as pd
import random
import numpy as np
from src.environment.job_dag import Job, Task


class JobManager:
    """管理作業資料的類別，負責從CSV加載數據並按順序提供作業"""
    
    def __init__(self):
        """初始化作業管理器"""
        self.all_jobs_data = None  # 儲存從CSV載入的所有作業數據
        self.current_index = 0     # 當前使用到的作業索引
        self.total_jobs = 0        # 總作業數量
    
    def load_all_jobs_from_csv(self, file_path):
        """
        一次性從CSV文件加載所有作業數據
        
        Args:
            file_path: CSV檔案路徑
        
        Returns:
            bool: 是否成功加載
        """
        try:
            # 讀取CSV檔案
            df = pd.read_csv(file_path)
            
            # 確保列名稱正確，處理列名稱可能的差異
            column_mapping = {
                'Task': 'Task',
                'Job id': 'Job id',
                'CPU': 'CPU',
                'MEM': 'MEM',
                'Disk': 'Disk'
            }
            
            # 檢查列名稱
            for expected_col, actual_col in column_mapping.items():
                if actual_col not in df.columns:
                    # 嘗試找到相似的列名（處理可能的空格或大小寫差異）
                    possible_columns = [col for col in df.columns if col.strip().lower() == actual_col.lower()]
                    if possible_columns:
                        df.rename(columns={possible_columns[0]: actual_col}, inplace=True)
                    else:
                        print(f"警告: 找不到'{actual_col}'列。請確認CSV檔案格式是否正確。")
            
            # 儲存所有作業數據
            self.all_jobs_data = df
            self.total_jobs = len(df['Job id'].unique())
            self.current_index = 0
            
            print(f"已從CSV成功載入 {self.total_jobs} 個唯一作業，共 {len(df)} 個任務")
            return True
            
        except Exception as e:
            print(f"載入CSV檔案時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_next_batch_jobs(self, env, current_time, batch_size=100):
        """
        獲取下一批作業數據並添加到環境中
        
        Args:
            env: 環境對象
            current_time: 當前時間
            batch_size: 批次大小
        
        Returns:
            int: 實際添加的作業數量
        """
        if self.all_jobs_data is None:
            print("錯誤: 尚未載入CSV數據")
            return 0
        
        # 獲取所有唯一的job_id
        unique_job_ids = self.all_jobs_data['Job id'].unique()
        
        # 確定本批次要處理的作業範圍
        start_idx = self.current_index
        end_idx = min(start_idx + batch_size, len(unique_job_ids))
        
        # 如果已經處理完所有作業，重新從頭開始
        if start_idx >= len(unique_job_ids):
            print("已處理完所有作業，重新從頭開始")
            start_idx = 0
            end_idx = min(batch_size, len(unique_job_ids))
            self.current_index = 0
        
        # 取得本批次的作業ID
        batch_job_ids = unique_job_ids[start_idx:end_idx]
        
        # 為每個作業创建Job對象並添加到環境中
        for job_id in batch_job_ids:
            # 创建一個新的Job對象
            job = Job(job_id, current_time)
            
            # 過濾出屬於當前job_id的所有任務
            job_tasks = self.all_jobs_data[self.all_jobs_data['Job id'] == job_id]
            
            # 為每個任務創建一個Task對象
            for _, task_row in job_tasks.iterrows():
                # 使用Task列的值作為任務ID的一部分
                task_name = task_row['Task']
                task_id = f"{job_id}_{task_name}"
                
                # 從CSV中讀取資源需求
                cpu = float(task_row['CPU'])
                ram = float(task_row['MEM'])
                disk = float(task_row['Disk'])
                
                # 創建Task對象
                task = Task(job_id=job_id, 
                            task_id=task_id,
                            CPU=cpu, 
                            RAM=ram, 
                            disk=disk, 
                            start_time=current_time)
                
                # 將任務添加到作業中
                job.add_task(task)
            
            # 將完整的Job添加到DAG中
            env.dag.add_job(job)
            env.total_jobs += 1
        
        # 更新當前索引
        self.current_index = end_idx
        
        # 返回實際添加的作業數量
        return len(batch_job_ids)
    
    def has_data(self):
        """
        檢查是否已加載數據
        
        Returns:
            bool: 是否有數據
        """
        return self.all_jobs_data is not None
    
    def reset_index(self):
        """重置當前索引"""
        self.current_index = 0


def generate_consistent_jobs(env, current_time, batch_size, random_seed=42):
    """
    使用固定的隨機種子生成 job，確保每次調用都生成相同的 job
    
    Args:
        env: 環境對象
        current_time: 當前時間
        batch_size: 要生成的 job 數量
        random_seed: 固定的隨機種子
    """
    # 暫時保存當前的隨機狀態
    current_state = random.getstate()
    random.seed(random_seed)
    np_current_state = np.random.get_state()
    np.random.seed(random_seed)
    
    for _ in range(batch_size):
        job_id = f"job_{env.total_jobs}"
        job = Job(job_id, current_time)
        num_tasks = 8
        for i in range(num_tasks):
            task = Task(job_id=job_id, task_id=f"{job_id}_task_{i}",
            CPU = random.uniform(0.01, 0.05), 
            RAM = random.uniform(0.01, 0.05),
            disk = random.uniform(0.01, 0.05), 
            start_time=current_time)
            job.add_task(task)
        env.dag.add_job(job)        
        env.total_jobs += 1
    
    # 恢復之前的隨機狀態
    random.setstate(current_state)
    np.random.set_state(np_current_state)


def generate_jobs_from_csv_or_random(env, current_time, jobs_per_episode, random_seed=42, csv_file_path=None, job_manager=None):
    """
    根據CSV文件或隨機生成作業
    
    Args:
        env: 環境對象
        current_time: 當前時間
        jobs_per_episode: 如果使用隨機生成，要生成的作業數量
        random_seed: 隨機種子
        csv_file_path: CSV文件路徑，如果提供則從CSV加載數據
        job_manager: JobManager實例
    """
    # 如果job_manager沒有加載數據且提供了CSV路徑，嘗試加載
    if csv_file_path and job_manager and not job_manager.has_data():
        success = job_manager.load_all_jobs_from_csv(csv_file_path)
        if not success:
            print("從CSV加載失敗，改用隨機生成...")
            generate_consistent_jobs(env, current_time, jobs_per_episode, random_seed)
            return
    
    # 如果已經加載了CSV數據，則從中獲取下一批作業
    if job_manager and job_manager.has_data():
        # 獲取下一批作業
        actual_jobs = job_manager.get_next_batch_jobs(env, current_time, jobs_per_episode)
        if actual_jobs < jobs_per_episode:
            generate_consistent_jobs(env, current_time, jobs_per_episode - actual_jobs, random_seed)
    else:
        # 如果沒有CSV數據，使用隨機生成
        generate_consistent_jobs(env, current_time, jobs_per_episode, random_seed)
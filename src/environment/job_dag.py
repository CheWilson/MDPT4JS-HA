
import random
from .virtual_clock import VirtualClock


class Task:
    """任務類 - 代表一個可執行的任務單元"""
    
    def __init__(self, job_id, task_id, CPU, RAM, disk, start_time):
        """
        初始化任務
        
        Args:
            job_id: 所屬作業ID
            task_id: 任務ID
            CPU: CPU資源需求
            RAM: 記憶體資源需求
            disk: 磁碟資源需求
            start_time: 開始時間
        """
        self.parent = []
        self.child = []
        self.job_id = job_id
        self.task_id = task_id
        self.CPU = float(CPU)
        self.RAM = float(RAM)
        self.disk = float(disk)
        self.status = 1  # 0: 完成, 1: 準備, 2: 運行中, 3: 拒絕
        self.start_time = start_time
        self.runtime = random.randint(25, 35)
        self.deadline = start_time + self.runtime + 1000
        self.endtime = None
        self.dc_i = None
        self.server_i = None
        self.ac_j = None
    
    def can_execute(self):
        """
        檢查任務是否可以執行（所有父任務都已完成）
        
        Returns:
            bool: 是否可以執行
        """
        return all(parent.status == 0 for parent in self.parent)
    
    def reject_with_children(self, env):
        """
        拒絕任務及其所有子任務
        
        Args:
            env: 環境對象
        """
        env.reject_related_tasks(self)


class Job:
    """作業類 - 包含多個相關任務的作業"""
    
    def __init__(self, job_id, creation_time):
        """
        初始化作業
        
        Args:
            job_id: 作業ID
            creation_time: 創建時間
        """
        self.job_id = job_id
        self.creation_time = creation_time
        self.tasks = []
        self.status = 1  # 1: 進行中, 0: 完成
        self.completion_time = None
    
    def add_task(self, task):
        """
        添加任務到作業中
        
        Args:
            task: 要添加的任務對象
        """
        self.tasks.append(task)
    
    def check_status(self):
        """
        檢查作業狀態
        
        Returns:
            int: 作業狀態 (0: 完成, 1: 進行中)
        """
        if all(task.status == 0 or task.status == 3 for task in self.tasks):
            self.status = 0
            completed_tasks = [task for task in self.tasks if task.status == 0 and task.endtime is not None]
            if completed_tasks and self.completion_time is None:
                self.completion_time = max(task.endtime for task in completed_tasks)
            elif not completed_tasks and self.completion_time is None:
                from_env_time = VirtualClock().time()
                self.completion_time = from_env_time
        return self.status


class DAG:
    """有向無環圖類 - 管理作業和任務之間的依賴關係"""
    
    def __init__(self):
        """初始化DAG"""
        self.jobs = {}
        self.ready_tasks = []
    
    def add_job(self, job):
        """
        添加作業到DAG中
        
        Args:
            job: 要添加的作業對象
        """
        self.jobs[job.job_id] = job
        self.generate_dependencies(job)
    
    def generate_dependencies(self, job):
        """
        為作業中的任務生成依賴關係
        
        Args:
            job: 作業對象
        """
        tasks = job.tasks
        n = len(tasks)
        if n <= 1:
            return
        
        shuffled_tasks = tasks.copy()
        random.shuffle(shuffled_tasks)
        
        for i in range(n-1):
            if random.random() < 0.7:
                current_task = shuffled_tasks[i]
                next_task = shuffled_tasks[i+1]
                next_task.parent.append(current_task)
                current_task.child.append(next_task)
    
    def update_ready_tasks(self, current_time):
        """
        更新準備執行的任務列表
        
        Args:
            current_time: 當前時間
        """
        self.ready_tasks = []
        for job in self.jobs.values():
            if job.status == 1:
                for task in job.tasks:
                    if (task.status == 1 and task.can_execute() and
                        task.start_time <= current_time and task not in self.ready_tasks):
                        self.ready_tasks.append(task)
    
    def update_job_status(self):
        """
        更新所有作業的狀態
        
        Returns:
            int: 已完成的作業數量
        """
        completed_jobs = 0
        for job in self.jobs.values():
            if job.check_status() == 0:
                completed_jobs += 1
        return completed_jobs
    
    def reset(self):
        """重置DAG"""
        self.jobs = {}
        self.ready_tasks = []
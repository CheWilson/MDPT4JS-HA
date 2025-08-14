import random
from .base_scheduler import BaseScheduler


class RandomFitScheduler(BaseScheduler):
    """Random Fit調度器 - 隨機選擇符合資源需求的位置分配任務"""
    
    def __init__(self, env):
        """
        初始化Random Fit調度器
        
        Args:
            env: 環境對象
        """
        super().__init__(env)
        self.name = "Random Fit"
    
    def schedule_task(self, task):
        """
        使用Random Fit策略調度任務
        
        Args:
            task: 要調度的任務
            
        Returns:
            tuple: (success, dc_id, server_id, ac_id) 或 (False, None, None, None)
        """
        current_time = self.env.virtual_clock.time()
        
        # 檢查任務截止期限
        if task.deadline < current_time + task.runtime:
            self.env.reject_related_tasks(task)
            self.env.rejected_by_deadline += 1
            return False, None, None, None
        
        # 收集所有可行的位置
        valid_positions = []
        for dc_id in range(self.env.num_dcs):
            # 檢查DC層級資源
            if not (self.env.dc_resources[dc_id][0] >= task.CPU and 
                    self.env.dc_resources[dc_id][1] >= task.RAM and 
                    self.env.dc_resources[dc_id][2] >= task.disk):
                continue
            
            for server_id in range(self.env.dc_size):
                # 檢查服務器層級資源
                if not (self.env.server_resources[dc_id][server_id][0] >= task.CPU and 
                        self.env.server_resources[dc_id][server_id][1] >= task.RAM and 
                        self.env.server_resources[dc_id][server_id][2] >= task.disk):
                    continue
                
                for ac_id in range(self.env.ac_per_server):
                    # 檢查AC層級資源
                    if self.env.check_resources(dc_id, server_id, ac_id, task):
                        valid_positions.append((dc_id, server_id, ac_id))
        
        # 如果有可行位置，隨機選擇一個
        if valid_positions:
            dc_id, server_id, ac_id = random.choice(valid_positions)
            
            # 記錄分配前的資源狀態
            dc_cpu_before = self.env.dc_resources[dc_id][0]
            server_cpu_before = self.env.server_resources[dc_id][server_id][0]
            ac_cpu_before = self.env.resources[dc_id][server_id][ac_id][0]
            
            # 執行任務分配
            success = self.env.allocate_task(task, dc_id, server_id, ac_id)
            if success:
                # 更新電力消耗
                self.env.calculate_power_reward(dc_id, server_id, ac_id, 
                                              dc_cpu_before, server_cpu_before, ac_cpu_before)
                return True, dc_id, server_id, ac_id
            else:
                # 分配失敗
                self.env.reject_related_tasks(task)
                self.env.rejected_tasks += 1
                return False, None, None, None
        else:
            # 沒有可行位置，拒絕任務
            self.env.reject_related_tasks(task)
            self.env.rejected_tasks += 1
            return False, None, None, None
    
    def get_description(self):
        """
        獲取調度器描述
        
        Returns:
            str: 調度器描述
        """
        return "Random Fit scheduler - randomly selects available resource positions for task allocation"
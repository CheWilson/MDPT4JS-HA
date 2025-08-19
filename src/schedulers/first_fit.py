from .base_scheduler import BaseScheduler


class FirstFitScheduler(BaseScheduler):
    """First Fit調度器 - 按順序找到第一個符合資源需求的位置分配任務"""
    
    def __init__(self, env):
        """
        初始化First Fit調度器
        
        Args:
            env: 環境對象
        """
        super().__init__(env)
        self.name = "First Fit"
    
    def schedule_task(self, task):
        """
        使用First Fit策略調度任務
        
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
        
        # First Fit策略: 找到第一個符合資源需求的位置
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
                        # 找到可行的位置，執行分配
                        dc_cpu_before = self.env.dc_resources[dc_id][0]
                        server_cpu_before = self.env.server_resources[dc_id][server_id][0]
                        ac_cpu_before = self.env.resources[dc_id][server_id][ac_id][0]
                        
                        success = self.env.allocate_task(task, dc_id, server_id, ac_id)
                        if success:
                            # 更新電力消耗
                            self.env.calculate_power_reward(dc_id, server_id, ac_id,
                                                          dc_cpu_before, server_cpu_before, ac_cpu_before)
                            return True, dc_id, server_id, ac_id
                        else:
                            # 分配失敗，繼續尋找下一個位置
                            continue
        
        # 沒有找到可行的位置，拒絕任務
        self.env.reject_related_tasks(task)
        self.env.rejected_tasks += 1
        return False, None, None, None
    
    def get_description(self):
        """
        獲取調度器描述
        
        Returns:
            str: 調度器描述
        """
        return "First Fit scheduler - sequentially finds the first available resource position for task allocation"
    
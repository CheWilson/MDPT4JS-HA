import numpy as np
from .virtual_clock import VirtualClock
from .job_dag import DAG


class Environment:
    """雲端任務調度環境類"""
    
    def __init__(self, total_servers, dc_size, ac_per_server=5):
        """
        初始化環境
        
        Args:
            total_servers: 總服務器數量
            dc_size: 每個資料中心的服務器數量
            ac_per_server: 每個服務器上的應用容器數量
        """
        self.virtual_clock = VirtualClock()
        self.dag = DAG()
        self.total_servers = total_servers
        self.dc_size = dc_size
        self.num_dcs = total_servers // dc_size
        self.ac_per_server = ac_per_server
        
        # 初始化資源狀態
        self._init_resources()
        
        # 初始化統計指標
        self._init_metrics()
        
        # 批次能耗記錄
        self.batch_power_records = []
        self.current_batch_power = []
        self.completed_job_count = 0
        self.batch_size = 100
    
    def _init_resources(self):
        """初始化資源狀態"""
        # AC層級資源 [dc][server][ac][resource_type]
        self.resources = [[[ [1.0/self.ac_per_server, 1.0/self.ac_per_server, 1.0/self.ac_per_server] 
                            for _ in range(self.ac_per_server)]
                          for _ in range(self.dc_size)]
                         for _ in range(self.num_dcs)]
        
        # DC層級資源 [dc][resource_type]
        self.dc_resources = [[self.dc_size, self.dc_size, self.dc_size] for _ in range(self.num_dcs)]
        
        # Server層級資源 [dc][server][resource_type]
        self.server_resources = [[[1.0, 1.0, 1.0] for _ in range(self.dc_size)] for _ in range(self.num_dcs)]
        
        # 任務分配記錄 [dc][server][ac]
        self.ac_tasks = [[[[] for _ in range(self.ac_per_server)] for _ in range(self.dc_size)]
                        for _ in range(self.num_dcs)]
    
    def _init_metrics(self):
        """初始化統計指標"""
        # 電力消耗
        self.dc_power = [0] * self.num_dcs
        self.server_power = [0] * self.total_servers
        self.ac_power = [0] * (self.total_servers * self.ac_per_server)
        self.total_power_consumption = 0
        self.power_per_time = []
        
        # 作業統計
        self.total_jobs = 0
        self.completed_jobs = 0
        
        # 任務統計
        self.total_tasks = 0
        self.succeeded_tasks = 0
        self.rejected_tasks = 0
        self.rejected_by_dc = 0
        self.rejected_by_server = 0
        self.rejected_by_ac = 0
        self.rejected_by_deadline = 0
    
    def get_power_consumption(self, remaining_resource, total_resource):
        """
        計算電力消耗
        
        Args:
            remaining_resource: 剩餘資源
            total_resource: 總資源
            
        Returns:
            float: 電力消耗
        """
        if remaining_resource > total_resource:
            remaining_resource = total_resource
        elif remaining_resource < 0:
            remaining_resource = 0
        
        static_power = 0.8 if remaining_resource < total_resource else 0 
        alpha = 0.5
        beta = 50
        utilization = (total_resource - remaining_resource) / total_resource
        
        if utilization < 0.7:
            dynamic_power = alpha * utilization
        else:
            dynamic_power = 0.7 * alpha + (utilization - 0.7)**2 * beta
        
        return dynamic_power + static_power
    
    def elecPrice(self, pwr):
        """
        計算電費
        
        Args:
            pwr: 電力消耗
            
        Returns:
            float: 電費
        """
        threshold = 1.5
        if pwr > 0 and pwr < threshold:
            p = 5.91
        elif pwr > 0 and pwr >= threshold:
            p = 8.27
        else:
            p = 1
        return pwr * p
    
    def check_resources(self, dc_id, server_id, ac_id, task):
        """
        檢查指定AC是否有足夠資源
        
        Args:
            dc_id: 資料中心ID
            server_id: 服務器ID
            ac_id: 應用容器ID
            task: 任務對象
            
        Returns:
            bool: 是否有足夠資源
        """
        cpu_avail = self.resources[dc_id][server_id][ac_id][0]
        ram_avail = self.resources[dc_id][server_id][ac_id][1]
        disk_avail = self.resources[dc_id][server_id][ac_id][2]
        return cpu_avail >= task.CPU and ram_avail >= task.RAM and disk_avail >= task.disk
    
    def check_dc_resources(self, task):
        """
        檢查哪些DC有足夠資源執行任務
        
        Args:
            task: 任務對象
            
        Returns:
            np.ndarray: DC可用性掩碼
        """
        mask = np.zeros(self.num_dcs, dtype=bool)
        for dc_id in range(self.num_dcs):
            # 先檢查DC層級資源是否足夠
            if (self.dc_resources[dc_id][0] >= task.CPU and 
                self.dc_resources[dc_id][1] >= task.RAM and 
                self.dc_resources[dc_id][2] >= task.disk):
            
                # 再檢查是否至少有一個伺服器可用
                has_viable_server = False
                for server_id in range(self.dc_size):
                    # 檢查伺服器資源
                    if (self.server_resources[dc_id][server_id][0] >= task.CPU and 
                        self.server_resources[dc_id][server_id][1] >= task.RAM and
                        self.server_resources[dc_id][server_id][2] >= task.disk):
                    
                        # 檢查是否至少有一個AC可用
                        has_viable_ac = False
                        for ac_id in range(self.ac_per_server):
                            if self.check_resources(dc_id, server_id, ac_id, task):
                                has_viable_ac = True
                                break
                    
                        if has_viable_ac:
                            has_viable_server = True
                            break
            
                # 只有當DC有足夠資源且至少有一個可用伺服器時，才標記為可用
                if has_viable_server:
                    mask[dc_id] = True
    
        return mask
    
    def check_server_resources(self, dc_id, task):
        """
        檢查指定DC中哪些服務器有足夠資源
        
        Args:
            dc_id: 資料中心ID
            task: 任務對象
            
        Returns:
            np.ndarray: 服務器可用性掩碼
        """
        mask = np.zeros(self.dc_size, dtype=bool)
        for server_id in range(self.dc_size):
            if (self.server_resources[dc_id][server_id][0] >= task.CPU and 
                self.server_resources[dc_id][server_id][1] >= task.RAM and
                self.server_resources[dc_id][server_id][2] >= task.disk):
                has_viable_ac = False
                for ac_id in range(self.ac_per_server):
                    if self.check_resources(dc_id, server_id, ac_id, task):
                        has_viable_ac = True
                        break
                if has_viable_ac:
                    mask[server_id] = True
        return mask
    
    def check_ac_resources(self, dc_id, server_id, task):
        """
        檢查指定服務器中哪些AC有足夠資源
        
        Args:
            dc_id: 資料中心ID
            server_id: 服務器ID
            task: 任務對象
            
        Returns:
            np.ndarray: AC可用性掩碼
        """
        mask = np.zeros(self.ac_per_server, dtype=bool)
        for ac_id in range(self.ac_per_server):
            if self.check_resources(dc_id, server_id, ac_id, task):
                mask[ac_id] = True
        return mask
    
    def allocate_task(self, task, dc_id, server_id, ac_id):
        """
        分配任務到指定資源
        
        Args:
            task: 任務對象
            dc_id: 資料中心ID
            server_id: 服務器ID
            ac_id: 應用容器ID
            
        Returns:
            bool: 分配是否成功
        """
        if not self.check_resources(dc_id, server_id, ac_id, task):
            return False
        
        current_time = self.virtual_clock.time()
        if task.deadline < current_time + task.runtime:
            return False
        
        # 分配資源
        self.resources[dc_id][server_id][ac_id][0] = round(self.resources[dc_id][server_id][ac_id][0] - task.CPU, 2)
        self.resources[dc_id][server_id][ac_id][1] = round(self.resources[dc_id][server_id][ac_id][1] - task.RAM, 2)
        self.resources[dc_id][server_id][ac_id][2] = round(self.resources[dc_id][server_id][ac_id][2] - task.disk, 2)
        
        self.server_resources[dc_id][server_id][0] = round(self.server_resources[dc_id][server_id][0] - task.CPU, 2)
        self.server_resources[dc_id][server_id][1] = round(self.server_resources[dc_id][server_id][1] - task.RAM, 2)
        self.server_resources[dc_id][server_id][2] = round(self.server_resources[dc_id][server_id][2] - task.disk, 2)
        
        self.dc_resources[dc_id][0] = round(self.dc_resources[dc_id][0] - task.CPU, 2)
        self.dc_resources[dc_id][1] = round(self.dc_resources[dc_id][1] - task.RAM, 2)
        self.dc_resources[dc_id][2] = round(self.dc_resources[dc_id][2] - task.disk, 2)
        
        # 更新任務狀態
        task.status = 2
        task.endtime = current_time + task.runtime
        task.dc_i = dc_id
        task.server_i = server_id
        task.ac_j = ac_id
        
        self.succeeded_tasks += 1
        self.ac_tasks[dc_id][server_id][ac_id].append(task)
        return True
    
    def reject_related_tasks(self, task):
        """
        拒絕任務及其所有子任務
        
        Args:
            task: 要拒絕的任務
        """
        if task.status != 3:
            self.rejected_tasks += 1
            task.status = 3
        for child_task in task.child:
            if child_task.status != 3:
                self.reject_related_tasks(child_task)
    
    def calculate_power_reward(self, dc_id, server_id, ac_id, 
                             dc_cpu_before, server_cpu_before, ac_cpu_before):
        """
        計算電力消耗變化帶來的獎勵
        
        Args:
            dc_id: 資料中心ID
            server_id: 服務器ID
            ac_id: 應用容器ID
            dc_cpu_before: 分配前DC的CPU資源
            server_cpu_before: 分配前服務器的CPU資源
            ac_cpu_before: 分配前AC的CPU資源
            
        Returns:
            tuple: (dc_reward, server_reward, ac_reward)
        """
        dc_total = self.dc_size
        server_total = 1.0
        ac_total = 1.0 / self.ac_per_server
        
        current_dc_power = self.get_power_consumption(dc_cpu_before, dc_total)
        current_server_power = self.get_power_consumption(server_cpu_before, server_total)
        current_ac_power = self.get_power_consumption(ac_cpu_before, ac_total)
        
        new_dc_power = self.get_power_consumption(self.dc_resources[dc_id][0], dc_total)
        new_server_power = self.get_power_consumption(self.server_resources[dc_id][server_id][0], server_total)
        new_ac_power = self.get_power_consumption(self.resources[dc_id][server_id][ac_id][0], ac_total)

        dc_power_change = new_dc_power - current_dc_power 
        server_power_change = new_server_power - current_server_power 
        ac_power_change = new_ac_power - current_ac_power

        dc_reward = - dc_power_change * 10 
        server_reward = - server_power_change * 10
        ac_reward = - ac_power_change * 10

        self.dc_power[dc_id] = new_dc_power
        self.server_power[dc_id * self.dc_size + server_id] = new_server_power
        ac_idx = (dc_id * self.dc_size + server_id) * self.ac_per_server + ac_id
        self.ac_power[ac_idx] = new_ac_power
        
        return dc_reward, server_reward, ac_reward
    
    def release_completed_tasks(self):
        """釋放已完成的任務並回收資源"""
        current_time = self.virtual_clock.time()
        
        # 取得目前已完成的 job 數量
        old_completed_jobs = self.completed_jobs
        
        for dc_id in range(self.num_dcs):
            for server_id in range(self.dc_size):
                for ac_id in range(self.ac_per_server):
                    tasks_to_remove = []
                    for task in self.ac_tasks[dc_id][server_id][ac_id]:
                        if task.endtime is not None and task.endtime <= current_time:
                            # 保存更新前的資源狀態，用於計算電力變化
                            dc_cpu_before = self.dc_resources[dc_id][0]
                            server_cpu_before = self.server_resources[dc_id][server_id][0]
                            ac_cpu_before = self.resources[dc_id][server_id][ac_id][0]
                            
                            # 釋放資源
                            self.resources[dc_id][server_id][ac_id][0] = round(self.resources[dc_id][server_id][ac_id][0] + task.CPU, 2)
                            self.resources[dc_id][server_id][ac_id][1] = round(self.resources[dc_id][server_id][ac_id][1] + task.RAM, 2)
                            self.resources[dc_id][server_id][ac_id][2] = round(self.resources[dc_id][server_id][ac_id][2] + task.disk, 2)
                            self.server_resources[dc_id][server_id][0] = round(self.server_resources[dc_id][server_id][0] + task.CPU, 2)
                            self.server_resources[dc_id][server_id][1] = round(self.server_resources[dc_id][server_id][1] + task.RAM, 2)
                            self.server_resources[dc_id][server_id][2] = round(self.server_resources[dc_id][server_id][2] + task.disk, 2)
                            self.dc_resources[dc_id][0] = round(self.dc_resources[dc_id][0] + task.CPU, 2)
                            self.dc_resources[dc_id][1] = round(self.dc_resources[dc_id][1] + task.RAM, 2)
                            self.dc_resources[dc_id][2] = round(self.dc_resources[dc_id][2] + task.disk, 2)
                            
                            # 更新電力消耗
                            dc_total = self.dc_size
                            server_total = 1.0
                            ac_total = 1.0 / self.ac_per_server
                            
                            # 計算新的電力消耗
                            new_dc_power = self.get_power_consumption(self.dc_resources[dc_id][0], dc_total)
                            new_server_power = self.get_power_consumption(self.server_resources[dc_id][server_id][0], server_total)
                            new_ac_power = self.get_power_consumption(self.resources[dc_id][server_id][ac_id][0], ac_total)
                            
                            # 更新電力消耗值
                            self.dc_power[dc_id] = new_dc_power
                            self.server_power[dc_id * self.dc_size + server_id] = new_server_power
                            ac_idx = (dc_id * self.dc_size + server_id) * self.ac_per_server + ac_id
                            self.ac_power[ac_idx] = new_ac_power
                            
                            # 記錄當前批次的能耗變化
                            if len(self.current_batch_power) < self.batch_size:
                                # 記錄釋放資源後的能耗變化（應為負值，因為能耗應該減少）
                                power_change = new_dc_power - self.dc_power[dc_id]
                                self.current_batch_power.append(power_change)
                            
                            task.status = 0
                            tasks_to_remove.append(task)
                    
                    for task in tasks_to_remove:
                        self.ac_tasks[dc_id][server_id][ac_id].remove(task)
        
        # 更新 job 狀態
        self.completed_jobs = self.dag.update_job_status()
        
        # 如果有新的 job 完成，更新計數
        if self.completed_jobs > old_completed_jobs:
            new_completed = self.completed_jobs - old_completed_jobs
            self.completed_job_count += new_completed
            
            # 每當完成 batch_size 個 job，記錄這批 job 的能耗數據
            if self.completed_job_count >= self.batch_size:
                # 記錄當前批次的能耗數據
                self.batch_power_records.append(self.current_batch_power.copy())
                # 重置當前批次的能耗記錄和計數器
                self.current_batch_power = []
                self.completed_job_count = 0
    
    def get_state(self, task=None):
        """
        獲取環境狀態 - 全局狀態向量格式
        
        Args:
            task: 當前待分配的任務，如果為None則使用零向量
        
        Returns:
            np.ndarray: 環境狀態向量
        """
        # 收集所有AC的資源狀態 (CPU, RAM, 磁碟)
        ac_resources = []
        ac_task_counts = []
        
        for dc_id in range(self.num_dcs):
            for server_id in range(self.dc_size):
                for ac_id in range(self.ac_per_server):
                    # 添加AC的資源狀態
                    ac_resources.extend(self.resources[dc_id][server_id][ac_id])
                    # 添加AC上的任務數量
                    ac_task_counts.append(len(self.ac_tasks[dc_id][server_id][ac_id]))
        
        # 合併AC資源狀態和任務數量
        state_vector = []
        for i in range(len(ac_resources) // 3):
            # 對於每個AC，添加其3個資源值和任務數量
            state_vector.extend(ac_resources[i*3:(i+1)*3])
            state_vector.append(ac_task_counts[i])
        
        # 添加當前任務的資源需求
        if task is not None:
            task_demands = [task.CPU, task.RAM, task.disk]
        else:
            task_demands = [0.0, 0.0, 0.0]
        
        state_vector.extend(task_demands)
        
        # 轉換為numpy數組
        state = np.array(state_vector, dtype=np.float32)
        return state
    
    def get_state_1(self, task=None):
        """
        獲取環境狀態 - MDPT4JS格式 (包含遮罩信息)
        
        Args:
            task: 當前待分配的任務
        
        Returns:
            np.ndarray: 環境狀態矩陣 [num_acs, 8]
        """
        # 獲取所有 AC 的資源狀態
        ac_state = np.array(self.resources, dtype=np.float32)
        # 将 3D 張量重塑為 2D 矩陣: (num_dcs*dc_size*ac_per_server, 3)
        state = ac_state.reshape(self.num_dcs * self.dc_size * self.ac_per_server, 3)

        # 計算每個AC上的任務數量
        ac_task_counts = np.zeros((state.shape[0], 1), dtype=np.float32)
        for dc_id in range(self.num_dcs):
            for server_id in range(self.dc_size):
                for ac_id in range(self.ac_per_server):
                    idx = (dc_id * self.dc_size + server_id) * self.ac_per_server + ac_id
                    ac_task_counts[idx, 0] = len(self.ac_tasks[dc_id][server_id][ac_id])

        # 如果有任務，將任務資源需求添加到每個 AC 特徵中
        if task is not None:
            # 獲取任務的資源需求
            task_demands = np.array([task.CPU, task.RAM, task.disk], dtype=np.float32)
            # 對任務需求廣播，使其與 AC 數量匹配
            task_demands_expanded = np.tile(task_demands, (state.shape[0], 1))
            
            # 計算每個AC是否可以接受當前任務(生成AC遮罩)
            ac_masks = np.zeros((state.shape[0], 1), dtype=np.float32)
            
            for i in range(state.shape[0]):
                # 檢查該AC的所有資源是否滿足需求
                if (state[i, 0] >= task_demands[0] and  # CPU
                    state[i, 1] >= task_demands[1] and  # RAM
                    state[i, 2] >= task_demands[2]):    # 磁碟
                    ac_masks[i, 0] = 1.0
            
            # 將 AC 資源狀態、任務需求、遮罩信息和AC任務數量拼接在一起
            state = np.hstack((state, task_demands_expanded, ac_masks, ac_task_counts))
        else:
            # 如果沒有任務，添加全零向量
            zero_demands = np.zeros((state.shape[0], 3), dtype=np.float32)
            zero_masks = np.zeros((state.shape[0], 1), dtype=np.float32)
            state = np.hstack((state, zero_demands, zero_masks, ac_task_counts))

        return state
    
    def get_state_2(self, task=None):
        """
        獲取環境狀態 - ACT4JS格式 (不包含遮罩信息)
        
        Args:
            task: 當前待分配的任務
        
        Returns:
            np.ndarray: 環境狀態矩陣 [num_acs, 7]
        """
        # 獲取所有 AC 的資源狀態
        ac_state = np.array(self.resources, dtype=np.float32)
        # 将 3D 張量重塑為 2D 矩陣: (num_dcs*dc_size*ac_per_server, 3)
        state = ac_state.reshape(self.num_dcs * self.dc_size * self.ac_per_server, 3)

        # 計算每個AC上的任務數量
        ac_task_counts = np.zeros((state.shape[0], 1), dtype=np.float32)
        for dc_id in range(self.num_dcs):
            for server_id in range(self.dc_size):
                for ac_id in range(self.ac_per_server):
                    idx = (dc_id * self.dc_size + server_id) * self.ac_per_server + ac_id
                    ac_task_counts[idx, 0] = len(self.ac_tasks[dc_id][server_id][ac_id])

        # 如果有任務，將任務資源需求添加到每個 AC 特徵中
        if task is not None:
            # 獲取任務的資源需求
            task_demands = np.array([task.CPU, task.RAM, task.disk], dtype=np.float32)
            # 對任務需求廣播，使其與 AC 數量匹配
            task_demands_expanded = np.tile(task_demands, (state.shape[0], 1))
            
            # 將 AC 資源狀態、任務需求和AC任務數量拼接在一起
            state = np.hstack((state, task_demands_expanded, ac_task_counts))
        else:
            # 如果沒有任務，添加全零向量
            zero_demands = np.zeros((state.shape[0], 3), dtype=np.float32)
            state = np.hstack((state, zero_demands, ac_task_counts))

        return state
    
    def step(self, task, dc_action, server_action, ac_action):
        """
        執行一步動作 - 基礎版本
        
        Args:
            task: 要分配的任務
            dc_action: DC選擇
            server_action: 服務器選擇
            ac_action: AC選擇
            
        Returns:
            tuple: (next_state, rewards, success)
        """
        self.total_tasks += 1
        dc_id = dc_action
        server_id = server_action
        ac_id = ac_action
        
        dc_cpu_before = self.dc_resources[dc_id][0]
        server_cpu_before = self.server_resources[dc_id][server_id][0]
        ac_cpu_before = self.resources[dc_id][server_id][ac_id][0]
        
        success = self.allocate_task(task, dc_id, server_id, ac_id)
        if success:
            dc_reward, server_reward, ac_reward = self.calculate_power_reward(
                dc_id, server_id, ac_id, dc_cpu_before, server_cpu_before, ac_cpu_before)
            next_state = self.get_state(task)
            return next_state, (dc_reward, server_reward, ac_reward), success
        else:
            task.reject_with_children(self)
            next_state = self.get_state(task)
            return next_state, (-5, -5, -5), False
    
    def step_1(self, task, dc_action, server_action, ac_action):
        """
        執行一步動作 - MDPT4JS版本
        """
        self.total_tasks += 1
        dc_id = dc_action
        server_id = server_action
        ac_id = ac_action
        
        dc_cpu_before = self.dc_resources[dc_id][0]
        server_cpu_before = self.server_resources[dc_id][server_id][0]
        ac_cpu_before = self.resources[dc_id][server_id][ac_id][0]
        
        allocation_success = self.allocate_task(task, dc_id, server_id, ac_id)
        if allocation_success:
            dc_reward, server_reward, ac_reward = self.calculate_power_reward(
                dc_id, server_id, ac_id, dc_cpu_before, server_cpu_before, ac_cpu_before)
            next_state = self.get_state_1(task)
            return next_state, (dc_reward, server_reward, ac_reward), True
        else:
            self.reject_related_tasks(task)
            next_state = self.get_state_1(task)
            return next_state, (-5, -5, -5), False
    
    def step_2(self, task, dc_action, server_action, ac_action):
        """
        執行一步動作 - ACT4JS版本
        """
        self.total_tasks += 1
        dc_id = dc_action
        server_id = server_action
        ac_id = ac_action
        
        dc_cpu_before = self.dc_resources[dc_id][0]
        server_cpu_before = self.server_resources[dc_id][server_id][0]
        ac_cpu_before = self.resources[dc_id][server_id][ac_id][0]
        
        allocation_success = self.allocate_task(task, dc_id, server_id, ac_id)
        if allocation_success:
            dc_reward, server_reward, ac_reward = self.calculate_power_reward(
                dc_id, server_id, ac_id, dc_cpu_before, server_cpu_before, ac_cpu_before)
            next_state = self.get_state_2(task)
            return next_state, (dc_reward, server_reward, ac_reward), True
        else:
            self.reject_related_tasks(task)
            next_state = self.get_state_2(task)
            return next_state, (-5, -5, -5), False
    
    def reset(self):
        """重置環境到初始狀態"""
        self.virtual_clock.reset()
        self.dag.reset()
        self._init_resources()
        self._init_metrics()
        
        # 重置批次能耗記錄
        self.current_batch_power = []
        self.completed_job_count = 0
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import time
from PIL import Image
import io
import logging
import sys
from collections import defaultdict

# 修正導入 - 使用你的項目結構
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.reproducibility import set_reproducibility
from utils.model_utils import load_model
from src.environment.environment import Environment
from src.environment.virtual_clock import VirtualClock
from src.environment.job_dag import Job, Task, DAG
from src.agents.dqn_agent import DQNAgent
from src.agents.ac_agent import ACAgent
from src.agents.mdpt4js_agent import MultiDiscreteACT4JSAgent
from src.agents.mdpt4js_ha_agent import MultiDiscreteACT4JSPredAgent
from src.agents.act4js_agent import ACT4JSAgent
from src.data.job_manager import JobManager, generate_jobs_from_csv_or_random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - 修改為只生成100個job的可視化
TOTAL_DCS = ENV_CONFIG['total_servers'] // ENV_CONFIG['dc_size']  # 10
SERVERS_PER_DC = ENV_CONFIG['dc_size']                           # 50  
ACS_PER_SERVER = ENV_CONFIG['ac_per_server']                     # 2
TOTAL_JOBS = 100        # 固定100個jobs
TASKS_PER_JOB = 8       # 每個job包含8個任務
MAX_SIMULATION_TIME = 200  # 限制最大模擬時間
SAVE_PATH = "./visualization_results"

# Create save directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(f"{SAVE_PATH}/gifs"):
    os.makedirs(f"{SAVE_PATH}/gifs")

class TaskTracker:
    def __init__(self):
        self.task_executions = []
        self.power_over_time = []
        self.cumulative_power = []
        self.tasks_by_time = defaultdict(list)
        self.task_details = {}
    
    def record_task_allocation(self, task, dc_id, server_id, ac_id, current_time):
        """Record when a task is allocated to a resource"""
        task_info = {
            'task_id': task.task_id,
            'job_id': task.job_id,
            'dc_id': dc_id,
            'server_id': server_id,
            'ac_id': ac_id,
            'start_time': current_time,
            'end_time': current_time + task.runtime,
            'cpu_usage': task.CPU,
            'ram_usage': task.RAM,
            'disk_usage': task.disk,
            'runtime': task.runtime
        }
        
        self.task_executions.append(task_info)
        self.task_details[task.task_id] = task_info
        
        for t in range(current_time, current_time + task.runtime):
            self.tasks_by_time[t].append(task.task_id)
    
    def record_power(self, current_time, power_value):
        """Record power consumption at each time step"""
        self.power_over_time.append((current_time, power_value))
        
        prev_cumulative = 0 if not self.cumulative_power else self.cumulative_power[-1][1]
        self.cumulative_power.append((current_time, prev_cumulative + power_value))
    
    def get_power_at_time(self, time_point):
        """Get power consumption at a specific time point"""
        for t, power in self.power_over_time:
            if t == time_point:
                return power
        return 0
    
    def get_cumulative_power_at_time(self, time_point):
        """Get cumulative power consumption at a specific time point"""
        for t, power in self.cumulative_power:
            if t == time_point:
                return power
        return 0
    
    def to_dataframe(self):
        """Convert task execution records to a DataFrame"""
        return pd.DataFrame(self.task_executions)
    
    def save_to_csv(self, filename):
        """Save task execution records to a CSV file"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        logger.info(f"Saved task execution data to {filename}")

class FirstFitAgent:
    """First Fit算法代理類"""
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def choose_action(self, state, mask):
        """選擇第一個可用的動作"""
        for action in range(self.n_actions):
            if mask[action]:
                return action
        return 0

class RandomFitAgent:
    """Random Fit算法代理類"""
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def choose_action(self, state, mask):
        """隨機選擇可用的動作"""
        valid_actions = [i for i in range(self.n_actions) if mask[i]]
        if valid_actions:
            return random.choice(valid_actions)
        return 0

def choose_action_single(agent, state, action_level, action_mask, method):
    """為特定層級選擇單一動作 - 適配你的代理結構"""
    state_tensor = T.tensor(state, dtype=T.float32).to(agent.device)
    
    with T.no_grad():
        if method.upper() == 'MDPT4JS-HA':
            dists, _, _ = agent.network(state_tensor)
            dist = dists[action_level]
        elif method.upper() == 'MDPT4JS':
            dists, _ = agent.network(state_tensor)
            dist = dists[action_level]
        elif method.upper() == 'ACT4JS':
            if action_level == 0:
                dist, _ = agent.dc_net(state_tensor)
            elif action_level == 1:
                dist, _ = agent.server_net(state_tensor)
            else:
                dist, _ = agent.ac_net(state_tensor)
        
        mask_tensor = T.tensor(action_mask, dtype=T.bool).to(agent.device)
        if mask_tensor.dim() < dist.logits.dim():
            mask_tensor = mask_tensor.unsqueeze(0)
        
        logits = dist.logits.clone()
        logits[~mask_tensor] = -1e9
        
        from torch.distributions.categorical import Categorical
        masked_dist = Categorical(logits=logits)
        action = masked_dist.sample().item()
    
    return action

def tracked_step(env, task, dc_action, server_action, ac_action, tracker, current_time, method_name):
    """Modified step function that records task execution details"""
    env.total_tasks += 1
    dc_id = dc_action
    server_id = server_action
    ac_id = ac_action
    
    dc_cpu_before = env.dc_resources[dc_id][0]
    server_cpu_before = env.server_resources[dc_id][server_id][0]
    ac_cpu_before = env.resources[dc_id][server_id][ac_id][0]
    
    success = env.allocate_task(task, dc_id, server_id, ac_id)
    
    if success:
        tracker.record_task_allocation(task, dc_id, server_id, ac_id, current_time)
        
        dc_reward, server_reward, ac_reward = env.calculate_power_reward(
            dc_id, server_id, ac_id, dc_cpu_before, server_cpu_before, ac_cpu_before)
            
        # 根據方法選擇適當的get_state函數
        if method_name in ['MDPT4JS', 'MDPT4JS-HA']:
            next_state = env.get_state_1(task)
        elif method_name == 'ACT4JS':
            next_state = env.get_state_2(task)
        else:  # DQN, AC, First Fit, Random Fit
            next_state = env.get_state(task)
            
        return next_state, (dc_reward, server_reward, ac_reward), True
    else:
        env.reject_related_tasks(task)
        
        if method_name in ['MDPT4JS', 'MDPT4JS-HA']:
            next_state = env.get_state_1(task)
        elif method_name == 'ACT4JS':
            next_state = env.get_state_2(task)
        else:
            next_state = env.get_state(task)
            
        return next_state, (-5, -5, -5), False

def generate_frame(time_point, tasks_df, power_usage, cumulative_power, method_name):
    """Generate a frame visualizing task scheduling for a specific time point"""
    logger.info(f"Generating frame for {method_name} at time {time_point}...")
    
    active_tasks = tasks_df[
        (tasks_df['start_time'] <= time_point) &
        (tasks_df['end_time'] > time_point)
    ]
    
    new_tasks = tasks_df[tasks_df['start_time'] == time_point]
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    
    dc_bg_color = '#e8f4f8'
    server_bg_color = '#d6eaf8'
    ac_bg_color = '#aed6f1'
    border_color = '#2874a6'
    
    margin = 0.1
    dc_width = 2.8
    dc_spacing = 0.2
    server_height = 0.2
    servers_per_column = 25
    
    unique_jobs = tasks_df['job_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_jobs)))
    job_colors = {job_id: colors[i] for i, job_id in enumerate(unique_jobs)}
    
    total_width = TOTAL_DCS * dc_width + (TOTAL_DCS - 1) * dc_spacing + 2 * margin
    total_height = (servers_per_column * server_height) + 4 * margin
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    title_height = total_height + 0.5
    ax.add_patch(plt.Rectangle(
        (0, title_height - 0.4), total_width, 0.4,
        edgecolor=None, facecolor='#34495e', alpha=1.0
    ))
    
    method_title = f"{method_name} - Time: {time_point}"
    ax.text(0.2, title_height - 0.2, method_title, 
            fontsize=14, ha='left', va='center', color='white', fontweight='bold')
    
    power_info = f"Current Power: {power_usage:.2f} | Cumulative Power: {cumulative_power:.2f}"
    ax.text(total_width - 0.2, title_height - 0.2, power_info, 
            fontsize=12, ha='right', va='center', color='white')
    
    # Draw DCs and servers
    for dc_id in range(TOTAL_DCS):
        dc_left = margin + dc_id * (dc_width + dc_spacing)
        
        dc_active_tasks = active_tasks[active_tasks['dc_id'] == dc_id]
        dc_cpu_usage = dc_active_tasks['cpu_usage'].sum() if not dc_active_tasks.empty else 0
        dc_capacity = SERVERS_PER_DC
        dc_usage_pct = (dc_cpu_usage / dc_capacity) * 100
        
        ax.text(dc_left + dc_width/2, total_height - 0.2, 
                f'DC {dc_id} ({dc_usage_pct:.1f}% used)',
                fontsize=9, ha='center', va='center', fontweight='bold')
        
        ax.add_patch(plt.Rectangle(
            (dc_left, margin), dc_width, total_height - 2*margin,
            edgecolor=border_color, linewidth=1, facecolor=dc_bg_color, alpha=0.5
        ))
        
        for server_id in range(SERVERS_PER_DC):
            col = 0 if server_id < servers_per_column else 1
            row = server_id % servers_per_column
            
            server_left = dc_left + (col * (dc_width/2))
            server_bottom = margin + (servers_per_column - 1 - row) * server_height
            
            server_active_tasks = dc_active_tasks[dc_active_tasks['server_id'] == server_id]
            server_cpu_usage = server_active_tasks['cpu_usage'].sum() if not server_active_tasks.empty else 0
            server_capacity = 1.0
            server_usage_pct = (server_cpu_usage / server_capacity) * 100
            
            server_width = dc_width/2 - 0.05
            ax.add_patch(plt.Rectangle(
                (server_left + 0.02, server_bottom), server_width - 0.04, server_height - 0.02,
                edgecolor=border_color, linewidth=0.5, facecolor=server_bg_color, alpha=0.7,
                zorder=1
            ))
            
            ax.text(server_left + 0.02, server_bottom + server_height/2 + 0.02, 
                    f'S{server_id}',
                    fontsize=3, ha='left', va='center', fontweight='bold', zorder=3)
            
            if server_usage_pct > 0:
                ax.text(server_left + 0.02, server_bottom + server_height/2 - 0.02, 
                        f'({server_usage_pct:.1f}%)',
                        fontsize=2, ha='left', va='center', fontweight='bold', zorder=3)
            
            for ac_id in range(ACS_PER_SERVER):
                ac_height = (server_height - 0.02) / ACS_PER_SERVER - 0.01
                ac_top = server_bottom + 0.01 + (ac_id * (ac_height + 0.01))
                
                ac_left = server_left + 0.20
                ac_width = server_width - 0.35
                
                ac_active_tasks = server_active_tasks[server_active_tasks['ac_id'] == ac_id]
                ac_cpu_usage = ac_active_tasks['cpu_usage'].sum() if not ac_active_tasks.empty else 0
                ac_capacity = 1.0 / ACS_PER_SERVER
                ac_usage_pct = (ac_cpu_usage / ac_capacity) * 100
                
                ax.add_patch(plt.Rectangle(
                    (ac_left, ac_top), ac_width, ac_height,
                    edgecolor=border_color, linewidth=0.5, facecolor=ac_bg_color,
                    alpha=0.5 if ac_usage_pct > 0 else 0.2,
                    zorder=2
                ))
                
                if ac_usage_pct > 0:
                    ax.text(ac_left + ac_width + 0.02, ac_top + ac_height/2,
                            f'{ac_usage_pct:.1f}%',
                            fontsize=2, ha='left', va='center', color='black', fontweight='bold', zorder=5)
                
                if not ac_active_tasks.empty:
                    current_active_tasks = ac_active_tasks[
                        (ac_active_tasks['start_time'] <= time_point) &
                        (ac_active_tasks['end_time'] > time_point)
                    ]
                    
                    current_left = ac_left
                    
                    for _, task in current_active_tasks.iterrows():
                        job_id = task['job_id']
                        task_id = task['task_id']
                        
                        job_num = job_id.replace('job_', '')
                        task_num = task_id.split('_')[-1]
                        
                        task_label = f"J{job_num}T{task_num}"
                        task_color = job_colors.get(job_id, (1, 0, 0))
                        
                        task_cpu_ratio = task['cpu_usage'] / ac_capacity
                        task_width = ac_width * task_cpu_ratio
                        
                        ax.add_patch(plt.Rectangle(
                            (current_left, ac_top), task_width, ac_height,
                            edgecolor='#2c3e50', linewidth=0.5, facecolor=task_color,
                            alpha=0.8, zorder=4
                        ))
                        
                        if task_width > 0.04:
                            ax.text(current_left + task_width/2, ac_top + ac_height/2,
                                   task_label, fontsize=3, ha='center', va='center',
                                   color='black', fontweight='bold', zorder=5)
                        
                        current_left += task_width
    
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, title_height)
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    logger.info(f"Frame for {method_name} at time {time_point} generated.")
    return img

def create_gif(frames, output_path, duration=1000):
    """Create a GIF from a list of frames"""
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
            disposal=2
        )
        logger.info(f"Saved GIF to {output_path}")
        return True
    else:
        logger.error("No frames generated, GIF not saved")
        return False

def run_method_visualization(method_name, model_path="./models", csv_file_path=None, 
                           ac_config=2, episode_num=100):
    """
    為特定方法生成可視化
    
    Args:
        method_name: 方法名稱
        model_path: 模型路徑
        csv_file_path: CSV數據文件路徑
        ac_config: AC配置（2, 6, 10）
        episode_num: 要載入的模型episode編號
    """
    logger.info(f"Generating visualization for {method_name} with AC config {ac_config}...")
    
    # 設置隨機種子
    set_reproducibility(42)
    
    # 創建環境
    env = Environment(
        total_servers=TOTAL_DCS * SERVERS_PER_DC,
        dc_size=SERVERS_PER_DC,
        ac_per_server=ac_config  # 使用指定的AC配置
    )
    
    # 創建任務追蹤器
    tracker = TaskTracker()
    
    # 創建作業管理器
    job_manager = JobManager()
    if csv_file_path and os.path.exists(csv_file_path):
        job_manager.load_all_jobs_from_csv(csv_file_path)
    
    # 創建代理
    if method_name in ["First Fit", "Random Fit"]:
        if method_name == "First Fit":
            agents = [
                FirstFitAgent(n_actions=env.num_dcs),
                FirstFitAgent(n_actions=env.dc_size),
                FirstFitAgent(n_actions=env.ac_per_server)
            ]
        else:  # Random Fit
            agents = [
                RandomFitAgent(n_actions=env.num_dcs),
                RandomFitAgent(n_actions=env.dc_size),
                RandomFitAgent(n_actions=env.ac_per_server)
            ]
    else:
        # 載入訓練好的模型
        model_load_path = os.path.join(model_path, f"ac{ac_config}")
        
        if method_name in ['MDPT4JS', 'MDPT4JS-HA']:
            input_dims = 8
        elif method_name == 'ACT4JS':
            input_dims = 7
        else:  # DQN, AC
            input_dims = env.num_dcs * env.dc_size * env.ac_per_server * 4 + 3
        
        n_actions = [env.num_dcs, env.dc_size, env.ac_per_server]
        
        try:
            agents = load_model(input_dims, n_actions, method_name, episode_num, model_load_path, env)
            if method_name == 'DQN' and len(agents) > 1:
                for agent in agents:
                    agent.epsilon = 0.1  # 設置低探索率
        except Exception as e:
            logger.error(f"Failed to load {method_name} model: {e}")
            return None
    
    # 生成作業
    current_time = 0
    generate_jobs_from_csv_or_random(
        env, current_time, TOTAL_JOBS,
        random_seed=42, csv_file_path=csv_file_path,
        job_manager=job_manager
    )
    
    # 執行調度模擬 - 修改為更精確的100個job控制
    jobs_completed = 0
    max_time = MAX_SIMULATION_TIME  # 使用更短的最大時間
    
    print(f"開始執行 {method_name} 調度模擬...")
    print(f"目標: 完成 {TOTAL_JOBS} 個jobs")
    
    while jobs_completed < TOTAL_JOBS and current_time < max_time:
        current_time += 1
        env.virtual_clock.advance(1)
        
        # 記錄能耗
        total_power_at_time = sum(env.ac_power)
        tracker.record_power(current_time, total_power_at_time)
        
        # 釋放完成的任務
        env.release_completed_tasks()
        env.dag.update_ready_tasks(current_time)
        
        # 處理準備好的任務
        tasks_processed_this_time = 0
        while env.dag.ready_tasks and tasks_processed_this_time < 10:  # 限制每個時間點處理的任務數
            task = env.dag.ready_tasks.pop(0)
            tasks_processed_this_time += 1
            
            # 檢查截止期限
            if task.deadline < current_time + task.runtime:
                env.reject_related_tasks(task)
                env.rejected_by_deadline += 1
                continue
            
            # 根據方法類型選擇動作
            if method_name in ["DQN", "AC"]:
                state = env.get_state(task=task)
                dc_mask = env.check_dc_resources(task)
                if not np.any(dc_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_dc += 1
                    continue
                
                dc_action = agents[0].choose_action(state, dc_mask)
                
                server_mask = env.check_server_resources(dc_action, task)
                if not np.any(server_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_server += 1
                    continue
                
                server_action = agents[1].choose_action(state, server_mask)
                
                ac_mask = env.check_ac_resources(dc_action, server_action, task)
                if not np.any(ac_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_ac += 1
                    continue
                
                ac_action = agents[2].choose_action(state, ac_mask)
                
            elif method_name in ["MDPT4JS", "MDPT4JS-HA"]:
                state = env.get_state_1(task)
                dc_mask = env.check_dc_resources(task)
                if not np.any(dc_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_dc += 1
                    continue
                
                dc_action = choose_action_single(agents[0], state, 0, dc_mask, method_name)
                
                server_mask = env.check_server_resources(dc_action, task)
                if not np.any(server_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_server += 1
                    continue
                
                server_action = choose_action_single(agents[0], state, 1, server_mask, method_name)
                
                ac_mask = env.check_ac_resources(dc_action, server_action, task)
                if not np.any(ac_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_ac += 1
                    continue
                
                ac_action = choose_action_single(agents[0], state, 2, ac_mask, method_name)
                
            elif method_name == "ACT4JS":
                state = env.get_state_2(task)
                dc_mask = env.check_dc_resources(task)
                if not np.any(dc_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_dc += 1
                    continue
                
                dc_action = choose_action_single(agents[0], state, 0, dc_mask, method_name)
                
                server_mask = env.check_server_resources(dc_action, task)
                if not np.any(server_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_server += 1
                    continue
                
                server_action = choose_action_single(agents[0], state, 1, server_mask, method_name)
                
                ac_mask = env.check_ac_resources(dc_action, server_action, task)
                if not np.any(ac_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_ac += 1
                    continue
                
                ac_action = choose_action_single(agents[0], state, 2, ac_mask, method_name)
                
            elif method_name in ["First Fit", "Random Fit"]:
                dc_mask = env.check_dc_resources(task)
                if not np.any(dc_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_dc += 1
                    continue
                
                dc_action = agents[0].choose_action(None, dc_mask)
                
                server_mask = env.check_server_resources(dc_action, task)
                if not np.any(server_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_server += 1
                    continue
                
                server_action = agents[1].choose_action(None, server_mask)
                
                ac_mask = env.check_ac_resources(dc_action, server_action, task)
                if not np.any(ac_mask):
                    env.reject_related_tasks(task)
                    env.rejected_by_ac += 1
                    continue
                
                ac_action = agents[2].choose_action(None, ac_mask)
            
            # 執行任務分配
            _, rewards, success = tracked_step(
                env, task, dc_action, server_action, ac_action, tracker, current_time, method_name
            )
        
        # 更新作業狀態
        old_jobs_completed = jobs_completed
        jobs_completed = env.dag.update_job_status()
        
        # 顯示進度
        if jobs_completed > old_jobs_completed:
            progress = (jobs_completed / TOTAL_JOBS) * 100
            print(f"  時間 {current_time}: 已完成 {jobs_completed}/{TOTAL_JOBS} jobs ({progress:.1f}%)")
    
    print(f"模擬完成！最終狀態:")
    print(f"  完成jobs: {jobs_completed}/{TOTAL_JOBS}")
    print(f"  模擬時間: {current_time}")
    print(f"  成功任務: {env.succeeded_tasks}")
    print(f"  拒絕任務: {env.rejected_tasks}")
    
    # 生成可視化幀 - 限制幀數以避免GIF過大
    task_df = tracker.to_dataframe()
    frames = []
    
    if tracker.task_executions:
        # 計算實際的結束時間，但限制最大幀數
        actual_end_time = max(task["end_time"] for task in tracker.task_executions)
        max_frame_time = min(actual_end_time + 5, MAX_SIMULATION_TIME)  # 限制最大幀數
        
        print(f"生成可視化幀: 1 到 {int(max_frame_time)} (共 {int(max_frame_time)} 幀)")
        
        # 生成每個時間點的幀
        for t in range(1, int(max_frame_time) + 1):
            if t % 20 == 0:  # 每20幀顯示一次進度
                print(f"  生成幀 {t}/{int(max_frame_time)}")
                
            power_at_t = tracker.get_power_at_time(t)
            cumulative_power_at_t = tracker.get_cumulative_power_at_time(t)
            frame = generate_frame(t, task_df, power_at_t, cumulative_power_at_t, 
                                 f"{method_name} (AC={ac_config})")
            frames.append(frame)
        
        # 創建GIF - 調整持續時間使GIF不會太快
        output_filename = f"{method_name.lower().replace(' ', '_')}_ac{ac_config}_100jobs.gif"
        output_path = os.path.join(SAVE_PATH, "gifs", output_filename)
        
        print(f"創建GIF: {output_filename}")
        create_gif(frames, output_path, duration=300)  # 300ms每幀，適中的播放速度
    
    # 生成可視化幀
    task_df = tracker.to_dataframe()
    frames = []
    
    if tracker.task_executions:
        max_end_time = max(task["end_time"] for task in tracker.task_executions) + 5
        
        for t in range(1, int(max_end_time) + 1):
            power_at_t = tracker.get_power_at_time(t)
            cumulative_power_at_t = tracker.get_cumulative_power_at_time(t)
            frame = generate_frame(t, task_df, power_at_t, cumulative_power_at_t, 
                                 f"{method_name} (AC={ac_config})")
            frames.append(frame)
        
        # 創建GIF
        output_filename = f"{method_name.lower().replace(' ', '_')}_ac{ac_config}_visualization.gif"
        output_path = os.path.join(SAVE_PATH, "gifs", output_filename)
        create_gif(frames, output_path, duration=500)
        
        # 保存數據
        csv_filename = f"{method_name.lower().replace(' ', '_')}_ac{ac_config}_100jobs_execution.csv"
        power_filename = f"{method_name.lower().replace(' ', '_')}_ac{ac_config}_100jobs_power.csv"
        
        task_df.to_csv(os.path.join(SAVE_PATH, csv_filename), index=False)
        power_df = pd.DataFrame(tracker.power_over_time, columns=["time", "power"])
        power_df.to_csv(os.path.join(SAVE_PATH, power_filename), index=False)
        
        total_energy = sum(power for _, power in tracker.power_over_time)
        
        logger.info(f"{method_name} (AC={ac_config}) visualization completed!")
        logger.info(f"Total energy: {total_energy:.2f}")
        logger.info(f"Tasks succeeded: {env.succeeded_tasks}, rejected: {env.rejected_tasks}")
        
        return {
            "method": method_name,
            "ac_config": ac_config,
            "total_energy": total_energy,
            "succeeded_tasks": env.succeeded_tasks,
            "rejected_tasks": env.rejected_tasks,
            "gif_path": output_path
        }

def main():
    """主函數 - 生成所有方法的可視化"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dynamic visualizations for RL task scheduling methods')
    parser.add_argument('--model-path', type=str, default='./models', 
                       help='Path to trained model parameters')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file containing task data')
    parser.add_argument('--output-path', type=str, default='./visualization_results',
                       help='Path to save visualization results')
    parser.add_argument('--ac-config', type=int, default=2, choices=[2, 6, 10],
                       help='AC per server configuration')
    parser.add_argument('--episode', type=int, default=100,
                       help='Episode number to load for trained models')
    parser.add_argument('--methods', nargs='+', 
                       default=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                       help='Methods to visualize')
    
    args = parser.parse_args()
    
    # 更新全局保存路徑
    global SAVE_PATH
    SAVE_PATH = args.output_path
    
    # 創建輸出目錄
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(f"{SAVE_PATH}/gifs"):
        os.makedirs(f"{SAVE_PATH}/gifs")
    
    print(f"Model path: {args.model_path}")
    print(f"CSV file: {args.csv}")
    print(f"Output path: {args.output_path}")
    print(f"AC configuration: {args.ac_config}")
    print(f"Episode: {args.episode}")
    print(f"Methods: {args.methods}")
    
    results = []
    
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Generating visualization for {method}...")
        print(f"{'='*60}")
        
        try:
            result = run_method_visualization(
                method_name=method,
                model_path=args.model_path,
                csv_file_path=args.csv,
                ac_config=args.ac_config,
                episode_num=args.episode
            )
            
            if result:
                results.append(result)
                print(f" {method} visualization completed!")
                print(f"   Total energy: {result['total_energy']:.2f}")
                print(f"   Success rate: {result['succeeded_tasks']/(result['succeeded_tasks']+result['rejected_tasks'])*100:.1f}%")
                print(f"   GIF saved to: {result['gif_path']}")
            else:
                print(f" {method} visualization failed!")
                
        except Exception as e:
            print(f" Error generating visualization for {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成總結報告
    if results:
        print(f"\n{'='*60}")
        print("VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        
        comparison_data = []
        for result in results:
            total_tasks = result['succeeded_tasks'] + result['rejected_tasks']
            success_rate = result['succeeded_tasks'] / total_tasks * 100 if total_tasks > 0 else 0
            
            comparison_data.append({
                'Method': result['method'],
                'AC Config': result['ac_config'],
                'Total Energy': result['total_energy'],
                'Success Rate (%)': success_rate,
                'GIF Path': result['gif_path']
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{SAVE_PATH}/visualization_summary.csv", index=False)
        
        print(df.to_string(index=False))
        print(f"\nSummary saved to: {SAVE_PATH}/visualization_summary.csv")
        print(f"All GIFs saved to: {SAVE_PATH}/gifs/")
    
    return results

if __name__ == "__main__":
    main()
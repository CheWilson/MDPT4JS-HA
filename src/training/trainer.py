import numpy as np
import torch as T
from config import *
from src.agents.dqn_agent import DQNAgent
from src.agents.ac_agent import ACAgent
from src.data.job_manager import generate_jobs_from_csv_or_random
from utils.model_utils import save_model


class Trainer:
    """強化學習訓練器類"""
    
    def __init__(self):
        """初始化訓練器"""
        self.supported_methods = ['DQN', 'AC', 'ACT4JS', 'MDPT4JS', 'MDPT4JS-HA']
    
    def _get_method_config(self, method):
        """
        獲取指定方法的配置
        
        Args:
            method: 方法名稱
            
        Returns:
            dict: 方法配置
        """
        method_configs = {
            'DQN': DQN_CONFIG,
            'AC': AC_CONFIG,
            'ACT4JS': ACT4JS_CONFIG,
            'MDPT4JS': MDPT4JS_CONFIG,
            'MDPT4JS-HA': MDPT4JS_HA_CONFIG
        }
        return method_configs.get(method.upper(), {})
    
    def _create_agents(self, method, env):
        """
        創建指定方法的代理
        
        Args:
            method: 方法名稱
            env: 環境對象
            
        Returns:
            代理對象或代理元組
        """
        method = method.upper()
        config = self._get_method_config(method)
        
        if method == 'DQN':
            input_dims = env.num_dcs * env.dc_size * env.ac_per_server * 4 + 3
            dc_agent = DQNAgent(
                input_dims=input_dims,
                n_actions=env.num_dcs,
                **config
            )
            server_agent = DQNAgent(
                input_dims=input_dims,
                n_actions=env.dc_size,
                **config
            )
            ac_agent = DQNAgent(
                input_dims=input_dims,
                n_actions=env.ac_per_server,
                **config
            )
            return dc_agent, server_agent, ac_agent
            
        elif method == 'AC':
            input_dims = env.num_dcs * env.dc_size * env.ac_per_server * 4 + 3
            dc_agent = ACAgent(
                input_dims=input_dims,
                n_actions=env.num_dcs,
                **config
            )
            server_agent = ACAgent(
                input_dims=input_dims,
                n_actions=env.dc_size,
                **config
            )
            ac_agent = ACAgent(
                input_dims=input_dims,
                n_actions=env.ac_per_server,
                **config
            )
            return dc_agent, server_agent, ac_agent
            
        elif method == 'MDPT4JS':
            from src.agents.mdpt4js_agent import MultiDiscreteACT4JSAgent
            
            input_dims = 8
            n_actions_list = [env.num_dcs, env.dc_size, env.ac_per_server]
            total_acs = env.num_dcs * env.dc_size * env.ac_per_server
            
            agent = MultiDiscreteACT4JSAgent(
                input_dims=input_dims,
                n_actions_list=n_actions_list,
                total_acs=total_acs,
                **config
            )
            return agent,
            
        elif method == 'MDPT4JS-HA':
            from src.agents.mdpt4js_ha_agent import MultiDiscreteACT4JSPredAgent
            
            input_dims = 8
            n_actions_list = [env.num_dcs, env.dc_size, env.ac_per_server]
            total_acs = env.num_dcs * env.dc_size * env.ac_per_server
            
            agent = MultiDiscreteACT4JSPredAgent(
                input_dims=input_dims,
                n_actions_list=n_actions_list,
                total_acs=total_acs,
                **config
            )
            return agent,
            
        elif method == 'ACT4JS':
            from src.agents.act4js_agent import ACT4JSAgent
            
            input_dims = 7
            n_actions_list = [env.num_dcs, env.dc_size, env.ac_per_server]
            
            agent = ACT4JSAgent(
                input_dims=input_dims,
                n_actions_list=n_actions_list,
                **config
            )
            return agent,
        
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _choose_action_single(self, agent, state, action_level, action_mask, method):
        """
        為特定層級選擇單一動作
        
        Args:
            agent: 代理對象
            state: 狀態
            action_level: 動作層級 (0: DC, 1: Server, 2: AC)
            action_mask: 動作掩碼
            method: 方法名稱
            
        Returns:
            int: 選擇的動作
        """
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
            
            mask = T.tensor(action_mask, dtype=T.bool).to(agent.device)
            if mask.dim() < dist.logits.dim():
                mask = mask.unsqueeze(0)
            
            logits = dist.logits.clone()
            logits[~mask] = -1e9
            
            from torch.distributions.categorical import Categorical
            masked_dist = Categorical(logits=logits)
            action = masked_dist.sample().item()
        
        return action
    
    def _reset_environment(self, env):
        """重置環境到初始狀態"""
        env.reset()
    
    def _calculate_episode_stats(self, env, method, agents, episode, ep_losses):
        """
        計算並返回episode統計信息
        
        Args:
            env: 環境對象
            method: 方法名稱
            agents: 代理列表
            episode: episode編號
            ep_losses: 本episode的損失列表
            
        Returns:
            dict: 統計信息字典
        """
        ep_total_power = sum(env.power_per_time)
        ep_reward = -ep_total_power
        total_tasks = env.succeeded_tasks + env.rejected_tasks
        success_rate = env.succeeded_tasks / total_tasks * 100 if total_tasks > 0 else 0
        
        # 計算平均損失
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        
        stats = {
            'total_power': ep_total_power,
            'episode_reward': ep_reward,
            'completed_jobs': env.completed_jobs,
            'success_rate': success_rate,
            'succeeded_tasks': env.succeeded_tasks,
            'rejected_tasks': env.rejected_tasks,
            'total_tasks': total_tasks,
            'avg_loss': avg_loss,
            'num_loss_updates': len(ep_losses)
        }
        
        # 針對不同方法計算額外的損失統計
        if method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
            if len(agents) == 1:
                agent = agents[0]
                
                # MDPT4JS-HA的額外統計
                if method.upper() == 'MDPT4JS-HA' and hasattr(agent, 'actor_losses'):
                    recent_actor = agent.actor_losses[-5:] if len(agent.actor_losses) >= 5 else agent.actor_losses
                    recent_critic = agent.critic_losses[-5:] if len(agent.critic_losses) >= 5 else agent.critic_losses
                    recent_pred = agent.prediction_losses[-5:] if len(agent.prediction_losses) >= 5 else agent.prediction_losses
                    
                    stats['actor_loss'] = np.mean(recent_actor) if recent_actor else 0
                    stats['critic_loss'] = np.mean(recent_critic) if recent_critic else 0
                    stats['prediction_loss'] = np.mean(recent_pred) if recent_pred else 0
        
        elif method.upper() == 'ACT4JS':
            if len(agents) == 1:
                agent = agents[0]
                recent_dc = agent.dc_losses[-5:] if len(agent.dc_losses) >= 5 else agent.dc_losses
                recent_server = agent.server_losses[-5:] if len(agent.server_losses) >= 5 else agent.server_losses
                recent_ac = agent.ac_losses[-5:] if len(agent.ac_losses) >= 5 else agent.ac_losses
                
                stats['dc_loss'] = np.mean(recent_dc) if recent_dc else 0
                stats['server_loss'] = np.mean(recent_server) if recent_server else 0
                stats['ac_loss'] = np.mean(recent_ac) if recent_ac else 0
        
        else:  # DQN, AC
            dc_agent, server_agent, ac_agent = agents
            recent_dc = dc_agent.all_losses[-5:] if len(dc_agent.all_losses) >= 5 else dc_agent.all_losses
            recent_server = server_agent.all_losses[-5:] if len(server_agent.all_losses) >= 5 else server_agent.all_losses
            recent_ac = ac_agent.all_losses[-5:] if len(ac_agent.all_losses) >= 5 else ac_agent.all_losses
            
            stats['dc_loss'] = np.mean(recent_dc) if recent_dc else 0
            stats['server_loss'] = np.mean(recent_server) if recent_server else 0
            stats['ac_loss'] = np.mean(recent_ac) if recent_ac else 0
        
        return stats
    
    def _display_episode_stats(self, episode, method, stats, env, show_details=False):
        """
        顯示episode統計信息 - 改進版本
        
        Args:
            episode: episode編號
            method: 方法名稱
            stats: 統計信息字典
            show_details: 是否顯示詳細信息
        """
        # 基本信息 - 每個episode都顯示
        power_str = f"{stats['total_power']:.2f}"
        loss_str = f"{stats['avg_loss']:.4f}" if stats['avg_loss'] > 0 else "N/A"
        success_str = f"{stats['success_rate']:.1f}%"
        
        print(f" Ep{episode:3d} [{method}] | Power: {power_str:>8} | Loss: {loss_str:>8} | Success: {success_str:>6} | Jobs: {stats['completed_jobs']:3d}")
        
        # 每10個episode或最後一個episode顯示詳細信息
        if show_details or episode % 10 == 0:
            print(f"    ├─  Episode獎勵: {stats['episode_reward']:.2f}")
            print(f"    ├─  任務統計: 成功{stats['succeeded_tasks']}/拒絕{stats['rejected_tasks']}/總計{stats['total_tasks']}")
            print(f"    ├─  損失更新次數: {stats['num_loss_updates']}")
            
            # 顯示詳細損失信息
            if method.upper() == 'MDPT4JS-HA' and 'actor_loss' in stats:
                print(f"    ├─  詳細損失:")
                print(f"    │   ├─ Actor: {stats['actor_loss']:.4f}")
                print(f"    │   ├─ Critic: {stats['critic_loss']:.4f}")
                print(f"    │   └─ Prediction: {stats['prediction_loss']:.4f}")
            elif method.upper() == 'ACT4JS' and 'dc_loss' in stats:
                print(f"    ├─  網絡損失:")
                print(f"    │   ├─ DC: {stats['dc_loss']:.4f}")
                print(f"    │   ├─ Server: {stats['server_loss']:.4f}")
                print(f"    │   └─ AC: {stats['ac_loss']:.4f}")
            elif method.upper() in ['DQN', 'AC'] and 'dc_loss' in stats:
                print(f"    ├─  層級損失:")
                print(f"    │   ├─ DC: {stats['dc_loss']:.4f}")
                print(f"    │   ├─ Server: {stats['server_loss']:.4f}")
                print(f"    │   └─ AC: {stats['ac_loss']:.4f}")
            
            # 顯示拒絕原因分析
            print(f"    └─ 拒絕原因: DC({env.rejected_by_dc}) Server({env.rejected_by_server}) AC({env.rejected_by_ac}) Deadline({env.rejected_by_deadline})")
            print()  # 空行分隔
    
    def train_method(self, env, method, num_episodes, jobs_per_episode, 
                    save_path, csv_file_path=None, job_manager=None, verbose=False):
        """
        訓練指定的RL方法
        
        Args:
            env: 環境對象
            method: 方法名稱
            num_episodes: 訓練episode數量
            jobs_per_episode: 每個episode的job數量
            save_path: 模型保存路徑
            csv_file_path: CSV文件路徑
            job_manager: JobManager實例
            verbose: 是否顯示詳細訓練信息
            
        Returns:
            tuple: (episode_rewards, losses, agents)
        """
        print(f"\n{'='*80}")
        print(f" 開始訓練 {method} 方法")
        print(f" Episodes: {num_episodes} |  Jobs/Episode: {jobs_per_episode}")
        print(f"{'='*80}")
        
        # 創建代理
        agents = self._create_agents(method, env)
        
        # 初始化記錄
        episode_rewards = []
        all_losses = []
        
        # 訓練循環 - 移除tqdm以便更好地顯示自定義信息
        for ep in range(num_episodes):
            # 重置環境
            self._reset_environment(env)
            current_time = env.virtual_clock.time()
            
            # 生成作業
            generate_jobs_from_csv_or_random(
                env, current_time, jobs_per_episode,
                random_seed=42+ep, csv_file_path=csv_file_path,
                job_manager=job_manager
            )
            
            jobs_completed_in_episode = 0
            ep_losses = []
            
            while jobs_completed_in_episode < jobs_per_episode:
                current_time += 1
                env.virtual_clock.advance(1)
                
                # 記錄能耗
                total_power_at_time = sum(env.ac_power)
                env.current_batch_power.append(total_power_at_time)
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
                    
                    # 根據方法選擇動作並學習
                    if method.upper() in ['MDPT4JS', 'MDPT4JS-HA']:
                        loss = self._train_multi_discrete_method(env, task, agents[0], method)
                    elif method.upper() == 'ACT4JS':
                        loss = self._train_act4js_method(env, task, agents[0])
                    else:  # DQN, AC
                        loss = self._train_traditional_method(env, task, agents, method)
                    
                    if loss is not None:
                        ep_losses.append(loss)
                
                # 更新作業狀態
                jobs_completed_in_episode = env.dag.update_job_status()
                
                # 補充作業如果需要
                if env.total_jobs < jobs_per_episode:
                    generate_jobs_from_csv_or_random(
                        env, current_time, jobs_per_episode - env.total_jobs,
                        random_seed=42+ep, csv_file_path=csv_file_path,
                        job_manager=job_manager
                    )
            
            # 記錄本episode的結果
            stats = self._calculate_episode_stats(env, method, agents, ep + 1, ep_losses)
            episode_rewards.append(stats['episode_reward'])
            
            if ep_losses:
                all_losses.append(np.mean(ep_losses))
            
            # 確保批次數據被記錄
            if env.current_batch_power and env.completed_job_count > 0:
                env.batch_power_records.append(env.current_batch_power.copy())
            
            # 顯示統計信息 - 每個episode都顯示基本信息
            show_details = verbose or (ep + 1) % 10 == 0 or ep == num_episodes - 1
            self._display_episode_stats(ep + 1, method, stats, env, show_details)
            
            # 定期保存模型
            if (ep + 1) % 20 == 0 or ep == num_episodes - 1:
                self._save_agents(agents, method, ep + 1, save_path)
        
        # 訓練完成統計
        print(f"\n{'='*80}")
        print(f" {method} 訓練完成")
        print(f"{'='*80}")
        final_stats = self._calculate_episode_stats(env, method, agents, num_episodes, [])
        print(f" 最終統計：")
        print(f"   ├─ 總Episodes: {num_episodes}")
        print(f"   ├─ 最終能耗: {final_stats['total_power']:.2f}")
        print(f"   ├─ 平均Episode獎勵: {np.mean(episode_rewards):.2f}")
        print(f"   ├─ 最終任務成功率: {final_stats['success_rate']:.2f}%")
        print(f"   ├─ 總成功任務: {final_stats['succeeded_tasks']}")
        print(f"   └─ 總拒絕任務: {final_stats['rejected_tasks']}")
        
        return episode_rewards, all_losses, agents
    
    def _train_multi_discrete_method(self, env, task, agent, method):
        """訓練多離散動作方法 (MDPT4JS, MDPT4JS-HA)"""
        state = env.get_state_1(task)
        
        # 階段1: 選擇DC
        dc_mask = env.check_dc_resources(task)
        if not np.any(dc_mask):
            env.reject_related_tasks(task)
            env.rejected_by_dc += 1
            return None
        
        dc_id = self._choose_action_single(agent, state, 0, dc_mask, method)
        
        # 階段2: 選擇服務器
        server_mask = env.check_server_resources(dc_id, task)
        if not np.any(server_mask):
            env.reject_related_tasks(task)
            env.rejected_by_server += 1
            return None
        
        server_id = self._choose_action_single(agent, state, 1, server_mask, method)
        
        # 階段3: 選擇AC
        ac_mask = env.check_ac_resources(dc_id, server_id, task)
        if not np.any(ac_mask):
            env.reject_related_tasks(task)
            env.rejected_by_ac += 1
            return None
        
        ac_id = self._choose_action_single(agent, state, 2, ac_mask, method)
        
        # 執行動作
        next_state, rewards, success = env.step_1(task, dc_id, server_id, ac_id)
        if not success:
            return None
        
        # 學習
        _, _, ac_reward = rewards
        agent.store_transition(state, [dc_id, server_id, ac_id], ac_reward, next_state, False)
        
        # 根據方法決定何時學習
        if method.upper() == 'MDPT4JS-HA':
            if agent.should_learn():
                return agent.learn()
        else:  # MDPT4JS
            return agent.learn()
        
        return None
    
    def _train_act4js_method(self, env, task, agent):
        """訓練ACT4JS方法"""
        state = env.get_state_2(task)
        
        # 階段1: 選擇DC
        dc_mask = env.check_dc_resources(task)
        if not np.any(dc_mask):
            env.reject_related_tasks(task)
            env.rejected_by_dc += 1
            return None
        
        dc_id = self._choose_action_single(agent, state, 0, dc_mask, 'ACT4JS')
        
        # 階段2: 選擇服務器
        server_mask = env.check_server_resources(dc_id, task)
        if not np.any(server_mask):
            env.reject_related_tasks(task)
            env.rejected_by_server += 1
            return None
        
        server_id = self._choose_action_single(agent, state, 1, server_mask, 'ACT4JS')
        
        # 階段3: 選擇AC
        ac_mask = env.check_ac_resources(dc_id, server_id, task)
        if not np.any(ac_mask):
            env.reject_related_tasks(task)
            env.rejected_by_ac += 1
            return None
        
        ac_id = self._choose_action_single(agent, state, 2, ac_mask, 'ACT4JS')
        
        # 執行動作
        next_state, rewards, success = env.step_2(task, dc_id, server_id, ac_id)
        if not success:
            return None
        
        # 學習
        agent.store_transition(state, [dc_id, server_id, ac_id], rewards, next_state, False)
        return agent.learn()
    
    def _train_traditional_method(self, env, task, agents, method):
        """訓練傳統方法 (DQN, AC)"""
        dc_agent, server_agent, ac_agent = agents
        
        state = env.get_state(task=task)
        
        # 階段1: 選擇DC
        dc_mask = env.check_dc_resources(task)
        if not np.any(dc_mask):
            env.reject_related_tasks(task)
            env.rejected_by_dc += 1
            return None
        
        dc_action = dc_agent.choose_action(state, dc_mask)
        dc_id = dc_action
        
        # 階段2: 選擇服務器
        server_mask = env.check_server_resources(dc_id, task)
        if not np.any(server_mask):
            env.reject_related_tasks(task)
            env.rejected_by_server += 1
            return None
        
        server_action = server_agent.choose_action(state, server_mask)
        server_id = server_action
        
        # 階段3: 選擇AC
        ac_mask = env.check_ac_resources(dc_id, server_id, task)
        if not np.any(ac_mask):
            env.reject_related_tasks(task)
            env.rejected_by_ac += 1
            return None
        
        ac_action = ac_agent.choose_action(state, ac_mask)
        
        # 執行動作
        next_state, rewards, success = env.step(task, dc_action, server_action, ac_action)
        if not success:
            return None
        
        # 學習
        dc_reward, server_reward, ac_reward = rewards
        next_state = env.get_state(task)
        
        dc_loss = dc_agent.learn(state, dc_action, dc_reward, next_state)
        server_loss = server_agent.learn(state, server_action, server_reward, next_state)
        ac_loss = ac_agent.learn(state, ac_action, ac_reward, next_state)
        
        return (dc_loss + server_loss + ac_loss) / 3.0
    
    def _save_agents(self, agents, method, episode, save_path):
        """保存代理模型"""
        if len(agents) == 1:
            # 單一代理 (MDPT4JS, MDPT4JS-HA, ACT4JS)
            agent = agents[0]
            import os
            method_dir = os.path.join(save_path, method.lower())
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            
            if hasattr(agent, 'network'):
                T.save(agent.network.state_dict(), 
                      os.path.join(method_dir, f'network_ep{episode}.pth'))
            elif hasattr(agent, 'dc_net'):  # ACT4JS
                T.save(agent.dc_net.state_dict(), 
                      os.path.join(method_dir, f'dc_network_ep{episode}.pth'))
                T.save(agent.server_net.state_dict(), 
                      os.path.join(method_dir, f'server_network_ep{episode}.pth'))
                T.save(agent.ac_net.state_dict(), 
                      os.path.join(method_dir, f'ac_network_ep{episode}.pth'))
        else:
            # 多個代理 (DQN, AC)
            dc_agent, server_agent, ac_agent = agents
            save_model(dc_agent, server_agent, ac_agent, method, episode, save_path)
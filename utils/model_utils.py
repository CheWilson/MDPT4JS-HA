import os
import torch as T
from src.agents.dqn_agent import DQNAgent
from src.agents.ac_agent import ACAgent


def save_model(dc_agent, server_agent, ac_agent, method, episode, save_path):
    """
    保存模型參數到指定路徑
    
    Args:
        dc_agent: 農場層代理
        server_agent: 服務器層代理
        ac_agent: 虛擬機層代理
        method: 使用的方法 ('dqn', 'ac', 'MDPT4JS', 'MDPT4JS-HA', 或 'ACT4JS')
        episode: 當前 episode 編號
        save_path: 保存路徑
    """
    # 確保保存路徑存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    method_dir = os.path.join(save_path, method.lower())
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)
    
    if method.lower() == 'dqn':
        # 保存DQN模型
        T.save(dc_agent.Q.state_dict(), os.path.join(method_dir, f'dc_Q_ep{episode}.pth'))
        T.save(dc_agent.Q_target.state_dict(), os.path.join(method_dir, f'dc_Q_target_ep{episode}.pth'))
        T.save(server_agent.Q.state_dict(), os.path.join(method_dir, f'server_Q_ep{episode}.pth'))
        T.save(server_agent.Q_target.state_dict(), os.path.join(method_dir, f'server_Q_target_ep{episode}.pth'))
        T.save(ac_agent.Q.state_dict(), os.path.join(method_dir, f'ac_Q_ep{episode}.pth'))
        T.save(ac_agent.Q_target.state_dict(), os.path.join(method_dir, f'ac_Q_target_ep{episode}.pth'))
        
    elif method.lower() == 'ac':
        # 保存AC模型
        T.save(dc_agent.actor.state_dict(), os.path.join(method_dir, f'dc_actor_ep{episode}.pth'))
        T.save(dc_agent.critic.state_dict(), os.path.join(method_dir, f'dc_critic_ep{episode}.pth'))
        T.save(server_agent.actor.state_dict(), os.path.join(method_dir, f'server_actor_ep{episode}.pth'))
        T.save(server_agent.critic.state_dict(), os.path.join(method_dir, f'server_critic_ep{episode}.pth'))
        T.save(ac_agent.actor.state_dict(), os.path.join(method_dir, f'ac_actor_ep{episode}.pth'))
        T.save(ac_agent.critic.state_dict(), os.path.join(method_dir, f'ac_critic_ep{episode}.pth'))
        
    elif method.lower() == 'mdpt4js':
        # 保存 MDPT4JS 模型
        T.save(dc_agent.network.state_dict(), os.path.join(method_dir, f'network_ep{episode}.pth'))
    
    elif method.lower() == 'mdpt4js-ha':
        # 保存 MDPT4JS-HA 模型
        T.save(dc_agent.network.state_dict(), os.path.join(method_dir, f'network_ep{episode}.pth'))
    
    elif method.lower() == 'act4js':
        # 保存 ACT4JS 模型 (三個獨立網路)
        T.save(dc_agent.dc_net.state_dict(), os.path.join(method_dir, f'dc_network_ep{episode}.pth'))
        T.save(dc_agent.server_net.state_dict(), os.path.join(method_dir, f'server_network_ep{episode}.pth'))
        T.save(dc_agent.ac_net.state_dict(), os.path.join(method_dir, f'ac_network_ep{episode}.pth'))
    
    print(f"已保存 {method.upper()} 模型參數到 {method_dir}，Episode {episode}")


def load_model(input_dims, n_actions, method, episode, load_path, env=None):
    """
    載入之前訓練好的模型
    
    Args:
        input_dims: 輸入維度
        n_actions: 動作空間大小或列表 [dc_actions, server_actions, ac_actions]
        method: 使用的方法 ('dqn', 'ac', 'MDPT4JS', 'MDPT4JS-HA', 或 'ACT4JS')
        episode: 要載入的 episode 編號
        load_path: 模型保存路徑
        env: 環境對象（用於獲取AC總數等信息）
    
    Returns:
        (dc_agent, server_agent, ac_agent): 載入參數後的代理元組
    """
    method_dir = os.path.join(load_path, method.lower())
    
    if method.lower() == 'dqn':
        dc_agent = DQNAgent(input_dims=input_dims, n_actions=n_actions[0], lr=0.01, gamma=0.99)
        server_agent = DQNAgent(input_dims=input_dims, n_actions=n_actions[1], lr=0.01, gamma=0.99)
        ac_agent = DQNAgent(input_dims=input_dims, n_actions=n_actions[2], lr=0.01, gamma=0.99)
        
        dc_agent.Q.load_state_dict(T.load(os.path.join(method_dir, f'dc_Q_ep{episode}.pth')))
        dc_agent.Q_target.load_state_dict(T.load(os.path.join(method_dir, f'dc_Q_target_ep{episode}.pth')))
        server_agent.Q.load_state_dict(T.load(os.path.join(method_dir, f'server_Q_ep{episode}.pth')))
        server_agent.Q_target.load_state_dict(T.load(os.path.join(method_dir, f'server_Q_target_ep{episode}.pth')))
        ac_agent.Q.load_state_dict(T.load(os.path.join(method_dir, f'ac_Q_ep{episode}.pth')))
        ac_agent.Q_target.load_state_dict(T.load(os.path.join(method_dir, f'ac_Q_target_ep{episode}.pth')))
        
    elif method.lower() == 'ac':
        dc_agent = ACAgent(input_dims=input_dims, n_actions=n_actions[0], lr_actor=0.01, lr_critic=0.01, gamma=0.9)
        server_agent = ACAgent(input_dims=input_dims, n_actions=n_actions[1], lr_actor=0.01, lr_critic=0.01, gamma=0.9)
        ac_agent = ACAgent(input_dims=input_dims, n_actions=n_actions[2], lr_actor=0.01, lr_critic=0.01, gamma=0.9)
        
        dc_agent.actor.load_state_dict(T.load(os.path.join(method_dir, f'dc_actor_ep{episode}.pth')))
        dc_agent.critic.load_state_dict(T.load(os.path.join(method_dir, f'dc_critic_ep{episode}.pth')))
        server_agent.actor.load_state_dict(T.load(os.path.join(method_dir, f'server_actor_ep{episode}.pth')))
        server_agent.critic.load_state_dict(T.load(os.path.join(method_dir, f'server_critic_ep{episode}.pth')))
        ac_agent.actor.load_state_dict(T.load(os.path.join(method_dir, f'ac_actor_ep{episode}.pth')))
        ac_agent.critic.load_state_dict(T.load(os.path.join(method_dir, f'ac_critic_ep{episode}.pth')))
        
    elif method.lower() == 'mdpt4js':
        # 載入 MDPT4JS 模型
        from src.agents.mdpt4js_agent import MultiDiscreteACT4JSAgent
        
        mdpt4js_dir = os.path.join(load_path, "mdpt4js")
        if os.path.exists(mdpt4js_dir) and os.path.exists(os.path.join(mdpt4js_dir, f'network_ep{episode}.pth')):
            method_dir = mdpt4js_dir
        
        input_dims = 8  # MDPT4JS 使用的狀態維度
        n_actions_list = n_actions
        total_acs = env.num_dcs * env.dc_size * env.ac_per_server if env else 1000
        
        agent = MultiDiscreteACT4JSAgent(
            input_dims=input_dims, 
            n_actions_list=n_actions_list,
            lr=1e-4, 
            gamma=0.9, 
            clip_epsilon=0.2,
            buffer_size=256,
            mini_batch_size=32,
            n_epochs=10,
            gae_lambda=0.95,
            entropy_coef=1e-2,
            total_acs=total_acs
        )
        agent.network.load_state_dict(T.load(os.path.join(method_dir, f'network_ep{episode}.pth')))
        print(f"已載入 MDPT4JS 模型參數，Episode {episode}")
        return agent, agent, agent  # 返回相同的agent三次，保持接口兼容
        
    elif method.lower() == 'mdpt4js-ha':
        # 載入 MDPT4JS-HA 模型
        from src.agents.mdpt4js_ha_agent import MultiDiscreteACT4JSPredAgent
        
        mdpt4js_ha_dir = os.path.join(load_path, "mdpt4js-ha")
        if os.path.exists(mdpt4js_ha_dir) and os.path.exists(os.path.join(mdpt4js_ha_dir, f'network_ep{episode}.pth')):
            method_dir = mdpt4js_ha_dir
        
        input_dims = 8  # MDPT4JS-HA 使用的狀態維度
        n_actions_list = n_actions
        total_acs = env.num_dcs * env.dc_size * env.ac_per_server if env else 1000
        
        agent = MultiDiscreteACT4JSPredAgent(
            input_dims=input_dims, 
            n_actions_list=n_actions_list,
            lr=1e-4, 
            gamma=0.9, 
            clip_epsilon=0.2,
            buffer_size=256,
            mini_batch_size=32,
            n_epochs=10,
            gae_lambda=0.95,
            entropy_coef=1e-2,
            total_acs=total_acs
        )
        agent.network.load_state_dict(T.load(os.path.join(method_dir, f'network_ep{episode}.pth')))
        print(f"已載入 MDPT4JS-HA 模型參數，Episode {episode}")
        return agent, agent, agent  # 返回相同的agent三次，保持接口兼容
        
    elif method.lower() == 'act4js':
        # 載入 ACT4JS 模型
        from src.agents.act4js_agent import ACT4JSAgent
        
        act4js_dir = os.path.join(load_path, "act4js")
        if os.path.exists(act4js_dir):
            method_dir = act4js_dir
        
        input_dims = 7  # ACT4JS 使用的狀態維度
        n_actions_list = n_actions
        
        agent = ACT4JSAgent(
            input_dims=input_dims, 
            n_actions_list=n_actions_list,
            lr=0.01, 
            gamma=0.9, 
            clip_epsilon=0.2,
            buffer_capacity=10000, 
            mini_batch_size=256
        )
        
        agent.dc_net.load_state_dict(T.load(os.path.join(method_dir, f'dc_network_ep{episode}.pth')))
        agent.server_net.load_state_dict(T.load(os.path.join(method_dir, f'server_network_ep{episode}.pth')))
        agent.ac_net.load_state_dict(T.load(os.path.join(method_dir, f'ac_network_ep{episode}.pth')))
        
        print(f"已載入 ACT4JS 模型參數，Episode {episode}")
        return agent, agent, agent
    
    print(f"已載入 {method.upper()} 模型參數，Episode {episode}")
    return dc_agent, server_agent, ac_agent


def create_model_directories(save_path, methods):
    """
    創建模型保存目錄
    
    Args:
        save_path: 基礎保存路徑
        methods: 方法列表
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for method in methods:
        method_dir = os.path.join(save_path, method.lower())
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)
            print(f"創建目錄: {method_dir}")


def get_model_info(model_path):
    """
    獲取模型目錄中的可用模型信息
    
    Args:
        model_path: 模型路徑
        
    Returns:
        dict: 包含各方法可用episode的字典
    """
    model_info = {}
    
    if not os.path.exists(model_path):
        return model_info
    
    for method_dir in os.listdir(model_path):
        method_path = os.path.join(model_path, method_dir)
        if os.path.isdir(method_path):
            episodes = []
            for file in os.listdir(method_path):
                if file.endswith('.pth'):
                    # 提取episode號碼
                    if 'ep' in file:
                        try:
                            ep_num = int(file.split('ep')[1].split('.')[0])
                            episodes.append(ep_num)
                        except ValueError:
                            continue
            
            if episodes:
                model_info[method_dir] = sorted(list(set(episodes)))
    
    return model_info
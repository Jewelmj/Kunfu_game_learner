from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent as PPO
from agents.linear_q_agent import LinearAgent as LINEAR

def get_agent_class(agent_type,on_policy_bool):
    """Selects the agent class based on the config AGENT_TYPE and the training policy."""
    if on_policy_bool and (agent_type == 'PPO') and PPO:
        return PPO
    elif not on_policy_bool and agent_type == 'DQN':
        return DQNAgent
    elif not on_policy_bool and agent_type == 'LINEAR' and LINEAR:
        return LINEAR
    else:
        raise ValueError(f"Unknown or unavailable AGENT_TYPE specified: {agent_type}. Must be 'DQN', 'PPO', or 'LINEAR'.")
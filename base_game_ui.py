import msvcrt
import time
from utils.wrapper import make_env_human
from config import ENV_ID, RENDER_MODE_HUMAN, STEP_DELAY, DEFAULT_ACTION

# --- Configuration ---
env_id = ENV_ID
RENDER_MODE = RENDER_MODE_HUMAN

# --- Action Mapping ---
key_to_action = {
    b'w': 1,    # UP
    b'a': 3,    # LEFT
    b'd': 2,    # RIGHT
    b's': 4,    # DOWN
    b'e': 7,    # RIGHTFIRE 
    b'q': 8,    # LEFTFIRE
    b' ': 0,    # NOOP (Spacebar)
}

try:
    env = make_env_human(env_id)
    observation, info = env.reset(seed=42)
    
    print(f"Environment '{env_id}' ready.")
    print(f"Observation Space (DQN Ready): {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.n}")
    print("\nPress keys (W, A, S, D, E, Q) to act, 'R' to reset, 'X' to quit.")

    terminated = False
    truncated = False

    action = DEFAULT_ACTION  # start with left
    
    while not (terminated or truncated):
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'x':
                print("\nQuitting game on user request.")
                break
            if key == b'r':
                print("\nResetting environment...")
                observation, info = env.reset()
                continue

            if key in key_to_action:
                action = key_to_action[key]
                print(f"-> Manual Action: Key '{key.decode()}' mapped to Action {action}")
            else:
                pass
        
        env.render() 
        observation, reward, terminated, truncated, info = env.step(action)
        
        if action != 0 or msvcrt.kbhit():
             step_info = info.get('episode', {})
             print(f"Step | Reward={reward:.1f}, Action taken: {action} | Episode Reward: {step_info.get('r', 'N/A')}")
             
        if terminated or truncated:
            print("Episode ended naturally. Resetting...")
            observation, info = env.reset()

        time.sleep(STEP_DELAY) 

finally:
    if 'env' in locals() and env:
        env.close()
        print("Game session ended.")

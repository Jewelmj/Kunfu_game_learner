import gymnasium as gym
import ale_py
import msvcrt
import time

# --- Configuration ---
env_id = 'ALE/KungFuMaster-v5'
RENDER_MODE = "human"

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

gym.register_envs(ale_py)

try:
    # 1. Create the environment
    env = gym.make(env_id, render_mode=RENDER_MODE)
    observation, info = env.reset(seed=42)
    print(f"Environment '{env_id}' ready. Press keys (W, A, S, D, E, Q) and hit Enter/Space to act, or press 'R' to reset, 'X' to quit.")

    terminated = False
    truncated = False

    action = 0  # start with NOOP (Action 0)
    
    while not (terminated or truncated):
        # 2. Check for keyboard input non-blockingly
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'x':
                print("\nQuitting game on user request.")
                break
            if key == b'r':
                print("\nResetting environment...")
                observation, info = env.reset()
                continue

            # Check for a recognized action key
            if key in key_to_action:
                action = key_to_action[key]
                print(f"-> Manual Action: Key '{key.decode()}' mapped to Action {action}")
            else:
                # If an unmapped key is pressed, do nothing (NOOP = 0)
                pass
        
        # 3. Take the step using the determined action (keyboard input or default 0)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)
        
        if action != 0 or msvcrt.kbhit():
             print(f"Step {info.get('lives', 'N/A')}: Reward={reward:.1f}, Action taken: {action}")

        if terminated or truncated:
            print("Episode ended naturally. Resetting...")
            observation, info = env.reset()

        # Essential: Slow down the loop slightly for playability and rendering updates
        time.sleep(0.1) 

finally:
    if 'env' in locals() and env:
        env.close()
        print("Game session ended.")
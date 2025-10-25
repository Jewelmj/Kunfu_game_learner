import gymnasium as gym
import ale_py
import msvcrt # Windows-specific library for non-blocking keyboard input
import time

# --- Configuration ---
env_id = 'ALE/KungFuMaster-v5'
RENDER_MODE = "human" # Set to "rgb_array" if the window doesn't show (see previous step)

# --- Action Mapping (Based on your documentation) ---
key_to_action = {
    b'w': 1,    # UP
    b'a': 3,    # LEFT
    b'd': 2,    # RIGHT
    b's': 4,    # DOWN
    b'e': 7,    # RIGHTFIRE (assuming E is the primary attack key/combo)
    b'q': 8,    # LEFTFIRE (assuming Q is the primary attack key/combo)
    b' ': 0,    # NOOP (Spacebar)
}
# --- End Configuration ---

gym.register_envs(ale_py)

try:
    # 1. Create the environment
    env = gym.make(env_id, render_mode=RENDER_MODE)
    observation, info = env.reset(seed=42)
    print(f"Environment '{env_id}' ready. Press keys (W, A, S, D, E, Q) and hit Enter/Space to act, or press 'R' to reset, 'X' to quit.")

    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = 0  # Default to NOOP (Action 0)
        
        # 2. Check for keyboard input non-blockingly
        if msvcrt.kbhit():
            key = msvcrt.getch()
            
            # Check for quit command
            if key == b'x':
                print("\nQuitting game on user request.")
                break
            
            # Check for reset command
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
        
        # Displaying output every 10 steps to prevent screen spam, or when an action is taken
        if action != 0 or msvcrt.kbhit():
             print(f"Step {info.get('lives', 'N/A')}: Reward={reward:.1f}, Action taken: {action}")


        # 4. Handle episode end
        if terminated or truncated:
            print("Episode ended naturally. Resetting...")
            observation, info = env.reset()

        # Essential: Slow down the loop slightly for playability and rendering updates
        time.sleep(0.05) 

finally:
    # 5. Clean up
    if 'env' in locals() and env:
        env.close()
        print("Game session ended.")
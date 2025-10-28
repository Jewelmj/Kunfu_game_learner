import imageio
import torch
import os
import webbrowser

def show_video_of_model(agent, env_factory_fn, seed=0, output_filename='video.mp4', fps=30):
    """
    Records an episode of the trained agent and saves it as an MP4 file.
    After saving, it attempts to open the video.

    Args:
        agent: The trained agent instance (must have an .act() method).
        env_factory_fn: A function that returns a preprocessed environment 
                          (like make_env_agent from wrapper.py).
        seed (int): Environment seed for reproducibility.
        output_filename (str): The name of the file to save the video to.
        fps (int): Frames per second for the output video.
    """
    env = env_factory_fn()
    
    try:
        raw_env = env.unwrapped.env
    except AttributeError:
        print("Error: Could not access the unwrapped environment. Assuming `env.render()` provides raw frames.")
        raw_env = env

    state, _ = env.reset(seed=seed)
    done = False
    frames = []

    print(f"Recording video to {output_filename}...")
    
    with torch.no_grad():
        while not done:
            raw_frame = raw_env.render()
            frames.append(raw_frame)
            
            action = agent.act(state, epsilon=0.0) 

            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
    env.close()

    try:
        imageio.mimsave(output_filename, frames, fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
        abs_path = os.path.abspath(output_filename)
        print(f"Video saved successfully to {abs_path}")
        
        webbrowser.open(abs_path)
        print(f"Attempting to open {abs_path} in your default player...")

    except Exception as e:
        print(f"Error saving or opening video (is ffmpeg installed?): {e}")


# Kungfu Game Learner
This repository contains custom Pytorch implimentation of RL algorythms(DQN, Lenier-QNet) to train an agent to play the Atari game Kung Fu Master.

## Installation and Setup:
### clone repo:
```bash
git clone https://github.com/Jewelmj/Kunfu_game_learner.git
cd Kunfu_game_learner
```
### conda setup:
```bash
conda create -n kfmaster_env python=3.12
conda activate kfmaster_env
pip install -r requirements.txt
```
### (Optional) Install Pytorch with GPU support:
> Only needed if you have an NVIDIA GPU and want GPU acceleration.
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
## Usage:
1. running the base game use this command(Manual Play):
> Play the Atari game manually to understand controls and environment dynamics.
```bash
python base_game_ui.py
```
2. Training Process:
> For training offpolicy agents like DQN and Lenier these diffrent agents. 
```bash
python train_offpolicy.py
```
> For training onpolicy agents like PPO these diffrent agents. 
```bash
python train_onpolicy.py
```
> note: change the config file with rigth agent name and hyper paremeters for training.
3. Generate a Video of a Trained Agent:
> For visualising how good the trained model is.
```bash
python create_video.py
```
> note: change the config file with rigth agent name and file paths.
3. Create training plots:
> For visualising the training process and performance.
```bash
python plot_log.py
```
> note: change the config file with rigth log file path.
## Project Structure:
Here’s the full folder and file layout:
```bash
Kunfu_game_learner/
├── base_game_ui.py
├── train_offpolicy.py
├── train_onpolicy.py
├── create_video.py
├── plot_log.py
├── models/ (Q-networks)
├── agents/ (DQN, Linear and PPo agents)
├── memory/ (Replay buffer, Rollout buffer)
├── utils/ (Wrappers, video, agent loader)
├── results/ (Saved models, videos, logs)
└── config.py
```
Here's the porpouse of each file:
1. base_game_ui.py: Manual Play Interface.
2. train_offpolicy.py: Main Training Loop for DQN and Linier-Qnet agents.
3. train_onpolicy.py: Main training loop for PPO agent.
4. create_video.py: Loads a trained model and generates a high-resolution gameplay video.
5. plot_log.py: plots the log files after training.
6. config.py: Stores all key parameters for the project.
7. agents/: Implements different agent classes such as DQNAgent, LinearAgent, etc.
8. models/: Contains different neural network architectures. (CNN for DQN)
9. results/: Stores logs, saved models, and recorded videos.
10. agents/dqn_agent.py: Implements the core DQN functions, act (epsilon-greedy action selection), step (storing experience in the buffer), and learn (calculating TD-target and performing optimization).
11. agents/linear_agent.py: similar to DQN without CNN layers.
12. models/cnn_q_network.py: Defines the Q_netwrok model with CNN and FC layers.
13. models/linear_q_network.py: the newral network for agents/linear_agent.py.
14. memory/replay_buffer.py: Implements the memory structure.
15. utils/wrapper.py: Define the gymnasium eviorment and do preprosing for the agent.
16. utils/vedio_utils: Contains functions for video recording, saving, and rendering.
17. utils/agent_selection: Central module to choose and initialize agents dynamically during training or evaluation.

## Environment and Methods:
1. Environment id: ALE/KungFuMaster-v5 
2. Algorithm: DQN with target network and experience  replay, Linear Agent (Simpler baseline model).
3. Preprocessing: Standard Atari preprocessing, including 4-frame skipping and stacking 4 consecutive frame, Life-based termination for stable episodic resets.
## Results and Performance:
Training logs and videos are automatically saved under:
```bash
results/
├── saved_models/
│   ├── kungfu_dqn_final.pth
│   └── kungfu_linear_final.pth
├── logs/
│   └── training_log.csv
└── video/
    └── evaluation_video.mp4
```
## 🎮 Demo Video
Watch the trained agent play Kung Fu Master
[![Watch the video]](https://youtube.com/shorts/GNXz_VnIido?feature=share>)

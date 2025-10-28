# Kunfu Game Learner
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
> For getting a feel for the game by playing manualy.
```bash
python base_game_ui.py
```
2. Training Process:
> For training these diffrent agents and save in saved_models folder. 
```bash
python train.py
```
> note: change the config file with rigth agent name and hyper paremeters for training.
3. Generate a Video of a Trained Agent:
> For visualising how good the trained model is.
```bash
python create_video.py
```
> note: change the config file with rigth agent name and hyper paremeters for training.
## Project Structure:
Here’s the full folder and file layout:
```bash
Kunfu_game_learner/
├── base_game_ui.py
├── train.py
├── create_video.py
├── models/ (Q-networks)
├── agents/ (DQN & Linear agents)
├── memory/ (Replay buffer)
├── utils/ (Wrappers, video, agent loader)
├── results/ (Saved models, videos, logs)
└── config.py
```
Here's the porpouse of each file:
1. base_game_ui.py: Manual Play Interface.
2. train.py: Main Training Loop.
3. create_video.py: Loads a trained model and generates a high-resolution gameplay video.
4. config.py: Stores all key parameters for the project.
5. agents/: Implements different agent classes such as DQNAgent, LinearAgent, etc.
6. models/: Contains different neural network architectures. (CNN for DQN)
7. results/: Stores logs, saved models, and recorded videos.
8. agents/dqn_agent.py: Implements the core DQN functions, act (epsilon-greedy action selection), step (storing experience in the buffer), and learn (calculating TD-target and performing optimization).
9. agents/linear_agent.py: similar to DQN without CNN layers.
10. models/cnn_q_network.py: Defines the Q_netwrok model with CNN and FC layers.
11. models/linear_q_network.py: the newral network for agents/linear_agent.py.
12. memory/replay_buffer.py: Implements the memory structure.
13. utils/wrapper.py: Define the gymnasium eviorment and do preprosing for the agent.
14. utils/vedio_utils: Contains functions for video recording, saving, and rendering.
15. utils/agent_selection: Central module to choose and initialize agents dynamically during training or evaluation.

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
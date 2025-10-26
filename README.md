# Kunfu Game Learner
This repository contains custom Pytorch implimentation of DQN algorythms to train an agent to play the Atari game Kung Fu Master.

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
```bash
python base_game_ui.py
```
2. Training Process:
```bash
python train.py
```
## Project Structure:
1. base_game_ui.py: Manual Play Interface.
2. train.py: Main Training Loop.
3. dqn_agent.py: Implements the core DQN functions, act (epsilon-greedy action selection), step (storing experience in the buffer), and learn (calculating TD-target and performing optimization).
4. q_network.py: Defines the Q_netwrok model with CNN and FC layers.
5. replay_buffer.py: Implements the memory structure.
6. wrapper.py: Define the gymnasium eviorment and do preprosing for the agent.
7. config.py: Stores all key parameters for the project.
## Environment and Methods:
1. Environment  id: ALE/KungFuMaster-v5 
2. Algorithm: DQN with target network and experience  replay.
3. Preprocessing: Standard Atari preprocessing, including 4-frame skipping and stacking 4 consecutive frames.
## Results and Performance:
(To be updated after the model has completed training.)


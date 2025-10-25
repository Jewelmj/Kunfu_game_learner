# Kunfu Game Learner

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
1. running the base game use this command:
```bash
python base_game_ui.py
```
## Environment and Methods:
1. env_id: ALE/KungFuMaster-v5 
## Results and Performance:
None
## Project Structure:
1. base_game_ui.py : the base game


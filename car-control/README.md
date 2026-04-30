# Parking RL
 
A 2D top-down reinforcement learning parking environment built with pygame.
 
## Folder structure
```
parking_rl/
├── parking_env.py   # Main environment + manual play
├── train.py         # Train a PPO agent (stable-baselines3)
├── models/          # Saved model checkpoints (.zip)
├── logs/            # Tensorboard training logs
└── README.md
```
 
## Setup
```bash
pip install pygame numpy stable-baselines3 gymnasium
```
 
## Run manually
```bash
python parking_env.py
```
Controls: W/S throttle/brake, A/D steer, R toggle reverse, SPACE reset
 
## Train RL agent
```bash
python train.py
```
Checkpoints saved to `models/` every 10,000 steps.
 
## Load a trained model in the environment
1. Run `python parking_env.py`
2. Click **Load Model** to cycle through saved models in `models/`
3. Click **Mode: HUMAN** to switch to **Mode: RL AGENT**
4. Watch the agent park!
## What was fixed
- Rays now cover full 360° evenly around the car heading
- Car must be fully stopped (speed < 0.05) to count as parked
- Reverse parking is accepted (car facing either direction into bay)
- Out of bounds correctly shows OUT OF BOUNDS, not PARKED
- RL agent mode button added
- Load model button cycles through saved .zip models

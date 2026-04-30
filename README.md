# 🤖 RL Projects

A collection of reinforcement learning experiments built to learn new skills

Each project explores a different RL environment — from classic control benchmarks to custom-built simulations from scratch.

---

## 📁 Projects

### 🚗 car-control
A custom 2D top-down parking simulation built from scratch using Pygame, trained with PPO (Stable-Baselines3).

- Built the entire environment from scratch — physics, ray sensors, reward shaping, UI
- Implemented **curriculum learning** across 5 progressive stages:
  - Stage 1 & 2 — Forward parking (straight and diagonal)
  - Stage 3 & 4 — Reverse parking (straight and diagonal)
  - Stage 5 — Parallel parking
-  Realistic Ackermann steering physics — car turns in proper arcs based on steering angle and wheelbase
- 12-ray sensor system for obstacle detection (simulating sensors in cars)
- In-app model loader and RL/human mode toggle
- Training monitored with **TensorBoard** — tracked reward curves, episode length, and policy metrics across multiple model iterations

**Stack:** Python, Pygame, Stable-Baselines3, NumPy, Gymnasium, TensorBoard

---

### 🐜 ant
Experimented with OpenAI Gym's Ant-v4 environment — a 3D quadruped robot learning to walk using RL in a physics simulation.

**Stack:** Python, Gymnasium, MuJoCo

---

### 🕹️ cartpole
Explored the classic CartPole-v1 control problem — balancing a pole on a moving cart using a well-trained RL agent.

**Stack:** Python, Gymnasium, Stable-Baselines3

---

## 🛠 Setup

```bash
pip install pygame numpy stable-baselines3 gymnasium
```

Each project folder contains its own instructions.

---

## 👤 Author
**Sineth Ranasinghe** — [GitHub](https://github.com/sineth-04) | [LinkedIn](https://www.linkedin.com/in/sineth-ranasinghe-4229a32a1)

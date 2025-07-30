# QMazeSolver

A simple yet powerful Q-learning implementation that teaches an agent to navigate a 5x8 maze while avoiding obstacles and optimizing its path to the goal. Designed for visual learners, this project includes real-time plots of reward progression and epsilon decay to help understand the learning dynamics.

---

## Features

- Q-learning algorithm from scratch
- 5×8 grid world with configurable blocked cells
- Start at (0,0), goal at (4,7)
- Blocked states simulate walls the agent can't pass
- Visualization of:
  - Total reward per episode
  - Moving average of performance
  - Epsilon decay over time

---

## Algorithms Used

- **Q-Learning** (model-free RL)
- **Epsilon-greedy exploration** with **decay strategy**
- **Moving average** to easier data virtualization

---

## Visual Output

| Rewards over Episodes | Epsilon Decay |
|------------------------|----------------|
| ![Rewards](img/rewards_plot.png) | ![Epsilon](img/epsilon_decay.png) |


This will:
1. Train a Q-learning agent over 500 episodes
2. Log total rewards and epsilon values
3. Plot learning curves using `matplotlib`

---

## Maze Layout

- Grid size: **5 rows × 8 columns**
- Blocked cells:
  ```
  (0,2), (1,2), (2,2), (3,2),
  (1,5), (2,5), (3,5), (4,5)
  ```

---

## Learning Parameters

| Parameter        | Value     |
|------------------|-----------|
| Episodes         | 500       |
| Learning rate α  | 0.1       |
| Discount γ       | 0.9       |
| Initial ε        | 1.0       |
| Min ε            | 0.01      |
| Decay rate       | 0.995     |

---

## License

MIT License — free to use, modify, and share.

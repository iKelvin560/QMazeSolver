import numpy as np
import matplotlib.pyplot as plt
import random

# Maze configuration
ROWS, COLS = 5, 8
ACTIONS = ['up', 'down', 'left', 'right']
START = (0, 0)
GOAL = (4, 7)
BLOCKED_STATES = {(0, 2), (1, 2), (2, 2), (3, 2), (1, 5), (2, 5), (3, 5), (4, 5)}

# Hyperparameters
ALPHA = 0.1      # Learning rate
GAMMA = 0.9      # Discount factor
EPISODES = 500

# Initialize decay strategy parameters
initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995
EPSILON = initial_epsilon
epsilon_history = []

def get_reward(state):
    if state == GOAL:
        return 100
    elif state in BLOCKED_STATES:
        return -100  # Blocked states should be unreachable
    else:
        return -1

def is_valid(state):
    row, col = state
    return (0 <= row < ROWS and 0 <= col < COLS and state not in BLOCKED_STATES)

def next_state(state, action):
    r, c = state
    if action == 'up':
        r -= 1
    elif action == 'down':
        r += 1
    elif action == 'left':
        c -= 1
    elif action == 'right':
        c += 1
    next_s = (r, c)
    return next_s if is_valid(next_s) else state

# Initialize Q-table
Q = {}
for row in range(ROWS):
    for col in range(COLS):
        pos = (row, col)
        if pos not in BLOCKED_STATES:
            Q[pos] = {a: 0 for a in ACTIONS}

# Training loop with reward tracking
episode_rewards = []

for episode in range(EPISODES):
    state = START
    total_reward = 0
    while state != GOAL:
        if random.uniform(0, 1) < EPSILON:
            action = random.choice(ACTIONS)
        else:
            action = max(Q[state], key=Q[state].get)

        new_state = next_state(state, action)
        reward = get_reward(new_state)
        best_next = max(Q[new_state].values()) if new_state in Q else 0

        Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
        total_reward += reward
        state = new_state

    episode_rewards.append(total_reward)
    epsilon_history.append(EPSILON)
    EPSILON = max(min_epsilon, EPSILON * decay_rate)

# Compute moving average
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plot rewards and epsilon
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Reward plot
ax1.plot(episode_rewards, color='lightgray', label='Total Reward per Episode')
ax1.plot(moving_average(episode_rewards), label='Moving Average (window=100)', color='blue')
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Q-Learning Agent: Total Rewards Across Episodes")
ax1.legend()
ax1.grid(True)

# Epsilon decay plot
ax2.plot(epsilon_history, label='Epsilon Value', color='green')
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.set_title("Epsilon Decay Over Episodes")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

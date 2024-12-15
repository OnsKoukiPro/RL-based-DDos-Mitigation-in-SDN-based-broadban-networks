import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions  # Available actions
        self.state_space_size = state_space_size  # Size of the state space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_space_size, len(actions)))  # Initialize Q-table

    def choose_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation

    def learn(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# Instantiate the Q-learning agent
actions = [0, 1]  # 0 for Normal, 1 for Attack
state_space_size = 1000  # Example size, you can adjust based on your features

agent = QLearningAgent(actions, state_space_size)

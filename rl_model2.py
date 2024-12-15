import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class RLAgent:
    def __init__(self, state_space_size, action_space, epsilon=1.0, alpha=0.1, gamma=0.9):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((state_space_size, len(action_space)))  # Initialize Q-table
        self.metrics = defaultdict(list)  # Store metrics during training
        self.reset_counters()

    def reset_counters(self):
        """Reset TP, TN, FP, FN counters."""
        self.tp = self.tn = self.fp = self.fn = 0

    def _state_to_index(self, state):
        """Map state tuple to index."""
        return hash(state) % self.state_space_size

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            state_index = self._state_to_index(state)
            return np.argmax(self.q_table[state_index])

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning formula."""
        state_index = self._state_to_index(state)
        next_state_index = self._state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.gamma * self.q_table[next_state_index, best_next_action]
        self.q_table[state_index, action] += self.alpha * (td_target - self.q_table[state_index, action])

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def learn(self, state, action, reward, next_state):
        """Wrapper for choosing action and updating Q-value."""
        self.update_q_value(state, action, reward, next_state)

    def track_metrics(self, action, attack_detected, is_attack):
        """Track TP, TN, FP, FN."""
        if action == 1:  # Block action
            if is_attack:
                self.tp += 1  # Correctly blocked attack
            else:
                self.fp += 1  # Incorrectly blocked normal traffic
        else:  # Allow action
            if is_attack:
                self.fn += 1  # Incorrectly allowed attack
            else:
                self.tn += 1  # Correctly allowed normal traffic

    def calculate_metrics(self):
        """Calculate accuracy, FPR, and FNR."""
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        fnr = self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0
        return accuracy, fpr, fnr

    def log_metrics(self, episode):
        """Log and store metrics after each episode."""
        accuracy, fpr, fnr = self.calculate_metrics()
        print(f"Episode {episode}: Accuracy={accuracy:.2f}, FPR={fpr:.2f}, FNR={fnr:.2f}")
        self.metrics["accuracy"].append(accuracy)
        self.metrics["fpr"].append(fpr)
        self.metrics["fnr"].append(fnr)

    def reset_metrics(self):
        """Reset counters for new tracking period."""
        self.reset_counters()



def get_real_time_traffic():
    """Retrieve real-time traffic data (SFE, SSIP, RFIP) and attack status."""
    # Placeholder function to simulate real-time traffic. Replace this with actual data retrieval.
    sfe = random.uniform(0, 10)  # Simulated SFE (e.g., packet size or session duration)
    ssip = random.uniform(0, 5)  # Simulated SSIP (e.g., source IP feature)
    rfip = random.uniform(0, 0.1)  # Simulated RFIP (e.g., response time feature)
    is_attack = random.choice([True, False])  # Randomly simulate attack traffic
    return (sfe, ssip, rfip), is_attack


# Training loop with continuous monitoring
if __name__ == "__main__":
    num_episodes = 1000
    state_space_size = 10000  # Size of the Q-table
    action_space = [0, 1]  # 0 = Allow, 1 = Block

    agent = RLAgent(state_space_size, action_space)

    for episode in range(num_episodes):
        agent.reset_counters()  # Reset counters for metrics tracking

        for step in range(100):  # Simulate 100 traffic flows per episode
            state, is_attack = get_real_time_traffic()  # Get real-time traffic and attack status
            action = agent.choose_action(state)  # Choose action based on state

            # Simulated reward function
            reward = 10 if (action == 1 and is_attack) else -2 if (action == 1 and not is_attack) else 5

            # Simulate next state (for this example, let's assume the next state is also retrieved from real-time traffic)
            next_state = get_real_time_traffic()[0]

            # Update Q-table with new state and reward
            agent.learn(state, action, reward, next_state)

            # Track metrics (True for attack, False for normal traffic)
            agent.track_metrics(action, is_attack, is_attack)

        # Calculate and log metrics after each episode
        agent.log_metrics(episode)

        # Decay exploration rate
        agent.decay_epsilon()

    # Visualize metrics
    plt.plot(agent.metrics["accuracy"], label="Accuracy")
    plt.plot(agent.metrics["fpr"], label="False Positive Rate")
    plt.plot(agent.metrics["fnr"], label="False Negative Rate")
    plt.xlabel("Episode")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("RL Agent Performance Metrics")
    plt.show()

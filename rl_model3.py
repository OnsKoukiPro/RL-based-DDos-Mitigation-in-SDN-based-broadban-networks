import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import logging

class DDPGAgent:
    def __init__(self, state_space_size, action_space_size, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, batch_size=64, buffer_size=1000000, theta=0.15, sigma=0.2):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = []
        self.buffer_size = buffer_size

        # Ornstein-Uhlenbeck noise parameters
        self.theta = theta  # Mean reversion
        self.sigma = sigma  # Volatility

        # Initialize networks
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Initialize noise
        self.action_noise = np.zeros(self.action_space_size)

        # Logging setup
        self.reward_history = []
        self.action_history = []

    # Ornstein-Uhlenbeck Noise (for stable exploration)
    def noise(self, action):
        noise = self.theta * (self.action_noise - action) + self.sigma * np.random.randn(self.action_space_size)
        self.action_noise = self.action_noise + noise  # Update the noise
        return np.clip(action + self.action_noise, -1.0, 1.0)

    # Add to replay buffer
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    # Sample from replay buffer
    def sample_from_replay_buffer(self):
        return random.sample(self.memory, self.batch_size)

    # Build Actor Network
    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_space_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_space_size, activation='tanh')
        ])
        return model

    # Build Critic Network
    def build_critic(self):
        state_input = layers.Input(shape=(self.state_space_size,))
        action_input = layers.Input(shape=(self.action_space_size,))
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Concatenate()([x, action_input])
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        return model

    # Train the agent
    def train(self):
        # Training logic: update the actor and critic networks using the replay buffer
        batch = self.sample_from_replay_buffer()
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Critic loss
        target_q = rewards + self.gamma * (1 - dones) * self.critic_target([next_states, self.actor_target(next_states)])
        with tf.GradientTape() as tape:
            critic_value = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q - critic_value))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Actor loss
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))  # Maximize the Q-value
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # Update the target networks
        self.update_target_networks()

    # Soft update of target networks
    def update_target_networks(self):
        for target, source in zip(self.actor_target.variables, self.actor.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        for target, source in zip(self.critic_target.variables, self.critic.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

    # Act: Select action based on state
    def act(self, state):
        state = np.reshape(state, (1, -1))
        action = self.actor(state)
        return self.noise(action)

    # Track performance (average reward and action distribution)
    def track_performance(self, reward, action):
        self.reward_history.append(reward)
        self.action_history.append(action)

        if len(self.reward_history) % 100 == 0:  # Log every 100 steps
            avg_reward = np.mean(self.reward_history[-100:])
            action_dist = np.mean(self.action_history[-100:])
            logging.info(f"Average Reward: {avg_reward:.2f}, Action Distribution: {action_dist:.2f}")

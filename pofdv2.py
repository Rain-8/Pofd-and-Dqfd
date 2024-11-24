import tensorflow as tf
import numpy as np
import pickle
from collections import deque
from Config import Config
from Memory import Memory

import numpy as np
import tensorflow as tf
from collections import deque

class POfDV2:
    def __init__(self, env, config, demo_transitions):
        self.env = env
        self.config = config
        self.gamma = config.GAMMA
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.demo_transitions = demo_transitions

        # Build policy and discriminator networks
        self._build_networks()

        # Initialize TensorFlow session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Pre-train the policy using demo data
        self.pre_train()

    def _build_networks(self):
        # Placeholders
        self.state_input = tf.compat.v1.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name="state")
        self.action_input = tf.compat.v1.placeholder(tf.int32, [None], name="action")
        self.advantage_input = tf.compat.v1.placeholder(tf.float32, [None], name="advantage")
        self.old_log_probs = tf.compat.v1.placeholder(tf.float32, [None], name="old_log_probs")
        self.disc_input = tf.compat.v1.placeholder(tf.float32, [None, self.env.observation_space.shape[0] + self.env.action_space.n], name="disc_input")
        self.disc_labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name="disc_labels")

        # Policy network
        with tf.compat.v1.variable_scope("policy_network"):
            hidden = tf.compat.v1.layers.dense(self.state_input, 128, activation=tf.nn.relu)
            self.policy_logits = tf.compat.v1.layers.dense(hidden, self.env.action_space.n, activation=None)
            self.policy_probs = tf.nn.softmax(self.policy_logits, axis=-1)
            action_masks = tf.one_hot(self.action_input, self.env.action_space.n)
            selected_action_probs = tf.reduce_sum(action_masks * self.policy_probs, axis=1)
            self.log_probs = tf.math.log(selected_action_probs + 1e-8)

        # PPO policy loss
        ratio = tf.exp(self.log_probs - self.old_log_probs)
        surrogate1 = ratio * self.advantage_input
        surrogate2 = tf.clip_by_value(ratio, 1 - self.config.EPSILON, 1 + self.config.EPSILON) * self.advantage_input
        # self.policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2)) - self.lambda2 * tf.reduce_mean(-tf.reduce_sum(self.policy_probs * tf.math.log(self.policy_probs + 1e-8), axis=1))
        # PPO policy loss with entropy regularization
        entropy = -tf.reduce_sum(self.policy_probs * tf.math.log(self.policy_probs + 1e-8), axis=1)
        self.policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2)) - self.lambda2 * tf.reduce_mean(entropy)


        # Discriminator network
        with tf.compat.v1.variable_scope("discriminator_network"):
            disc_hidden = tf.compat.v1.layers.dense(self.disc_input, 128, activation=tf.nn.relu)
            self.disc_output = tf.compat.v1.layers.dense(disc_hidden, 1, activation=tf.nn.sigmoid)
            self.disc_loss = -tf.reduce_mean(self.disc_labels * tf.math.log(self.disc_output + 1e-8) +
                                             (1 - self.disc_labels) * tf.math.log(1 - self.disc_output + 1e-8))

        # Optimizers
        self.policy_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.policy_loss)
        self.disc_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.disc_loss)

    def pre_train(self):
        print("Pre-training using demo data...")
        for step in range(self.config.PRETRAIN_STEPS):
            try:
                batch = self._sample_batch(self.demo_transitions, self.batch_size)
                states, actions, rewards = batch["states"], batch["actions"], batch["rewards"]
                discounted_rewards = self._compute_discounted_rewards(rewards)

                # Feed the pre-training data
                self.sess.run(self.policy_optimizer, feed_dict={
                    self.state_input: states,
                    self.action_input: actions,
                    self.advantage_input: discounted_rewards,
                    self.old_log_probs: np.zeros_like(discounted_rewards)  # Initialize to zeros for pretraining
                })
                if step % 200 == 0 and step > 0:
                    print('{} th step of pre-train finish ...'.format(step))
            except ValueError as e:
                print(f"Pre-training skipped at step {step}: {e}")
                break
        print("Pre-training completed.")

    def _sample_batch(self, buffer, batch_size):
        
        """
        Sample a batch of transitions from the buffer.

        Parameters:
            buffer (deque): Buffer of transitions.
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: Batch of sampled transitions.
        """
        # Flatten and validate buffer content
        buffer_list = [t for t in buffer if isinstance(t, (tuple, list)) and len(t) >= 3]
        if len(buffer_list) < batch_size:
            raise ValueError(f"Not enough valid samples in the buffer to create a batch of size {batch_size}. "
                            f"Buffer contains {len(buffer_list)} valid samples.")
        
        # Sample batch
        sampled_batch = np.random.choice(len(buffer_list), batch_size, replace=False)  # Indices of samples
        sampled_transitions = [buffer_list[i] for i in sampled_batch]
        
        # Unpack sampled transitions
        states, actions, rewards = zip(*[(t[0], t[1], t[2]) for t in sampled_transitions])
        return {"states": np.array(states), "actions": np.array(actions), "rewards": np.array(rewards)}


    def _compute_discounted_rewards(self, rewards):
        discounted = []
        cumulative = 0
        for r in reversed(rewards):
            cumulative = r + self.gamma * cumulative
            discounted.insert(0, cumulative)
        return np.array(discounted)*0.5

    def sample_action(self, state):
        policy_probs = self.sess.run(self.policy_probs, feed_dict={self.state_input: [state]})
        return np.random.choice(len(policy_probs[0]), p=policy_probs[0])

    def update_rewards_with_discriminator(self, trajectory):
        updated_trajectory = []
        for state, action, reward in trajectory:
            one_hot_action = np.eye(self.env.action_space.n)[action]
            state_action = np.concatenate([state, one_hot_action])
            disc_value = self.sess.run(self.disc_output, feed_dict={self.disc_input: [state_action]})[0][0]
            updated_reward = reward - self.lambda1 * np.log(disc_value + 1e-8)
            updated_trajectory.append((state, action, updated_reward))
        return updated_trajectory

    def train_discriminator(self, agent_trajectory, demo_transitions):
        """
        Train the discriminator using agent-generated and demo transitions.

        Parameters:
            agent_trajectory (list): A trajectory of agent-generated transitions.
            demo_transitions (deque): Expert demonstration transitions.
        """
        agent_data = []
        expert_data = []

        # Prepare agent-generated data
        for state, action, _ in agent_trajectory:
            one_hot_action = np.eye(self.env.action_space.n)[action]
            agent_data.append(np.concatenate([state, one_hot_action]))

        # Prepare expert demonstration data (extract only state, action, and reward)
        for transition in demo_transitions:
            state, action = transition[:2]  # Extract first two elements (state, action)
            one_hot_action = np.eye(self.env.action_space.n)[action]
            expert_data.append(np.concatenate([state, one_hot_action]))

        agent_labels = np.zeros((len(agent_data), 1))
        expert_labels = np.ones((len(expert_data), 1))
        x_data = np.vstack([agent_data, expert_data])
        y_data = np.vstack([agent_labels, expert_labels])

        # Train the discriminator
        self.sess.run(self.disc_optimizer, feed_dict={
            self.disc_input: x_data,
            self.disc_labels: y_data
        })

    def train_policy(self, trajectory):
        states, actions, rewards = zip(*trajectory)
        discounted_rewards = self._compute_discounted_rewards(rewards)
        old_log_probs = self.sess.run(self.log_probs, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })

        self.sess.run(self.policy_optimizer, feed_dict={
            self.state_input: states,
            self.action_input: actions,
            self.advantage_input: discounted_rewards,
            self.old_log_probs: old_log_probs
        })

    def sample_action(self, state):
        """
        Select an action using the current policy.

        Parameters:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        policy_probs = self.sess.run(self.policy_probs, feed_dict={self.state_input: [state]})[0]
        action = np.random.choice(len(policy_probs), p=policy_probs)
        return action




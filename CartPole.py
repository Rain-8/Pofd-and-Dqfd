# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfD_V3 import DQfD
from DQfDDDQN import DQfDDDQN
from pofdv2 import POfDV2
from collections import deque
import itertools
import argparse
import psutil
import time
from gym.wrappers import Monitor


# extend [n_step_reward, n_step_away_state] for transitions in demo
def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list



def get_demo_data(env):
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.variable_scope('get_demo_data'):
        agent = DQfDDDQN(env, DDQNConfig())

    e = 0
    while True:
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            demo.append([state, action, reward, next_state, done, 1.0])  # record the data that could be expert-data
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            if score == 500:  # expert demo data
                demo = set_n_step(demo, Config.trajectory_n)
                agent.demo_buffer.extend(demo)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)




def run_DDQN(index, env):
    with tf.variable_scope('DDQN_' + str(index)):
        agent = DQfDDDQN(env, DDQNConfig())
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
    return scores



def run_DQfD(index, env, demo_transitions=None):
    # Load pre-existing demo data from 'demo.p' file
    demo_path = Config.DEMO_DATA_PATH
    try:
        with open(demo_path, 'rb') as f:
            demo_transitions = pickle.load(f)
            demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
            assert len(demo_transitions) == Config.demo_buffer_size
            print(f"Loaded {len(demo_transitions)} demo transitions from {demo_path}.")
    except Exception as e:
        print(f"Error loading demonstrations: {e}")
        return

    # Initialize the agent with loaded demo data
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)

    # Measure memory and time
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB


    # Pre-train using only the loaded demo data
    agent.pre_train()
    print("Pre-training with demo data completed.")

    scores, e, replay_full_episode = [], 0, None
    early_stopping_counter = 0
    early_stopping_threshold = 3
    score_threshold = 500

    # Training loop
    while True:
        done, score, n_step_reward, state = False, 0, None, env.reset()
        t_q = deque(maxlen=Config.trajectory_n)

        while not done:
            action = agent.egreedy_action(state)  # E-greedy action for training
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]
            t_q.append([state, action, reward, next_state, done, 0.0])

            if len(t_q) == t_q.maxlen:
                if n_step_reward is None:
                    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_q)])
                else:
                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                    n_step_reward += reward * Config.GAMMA**(Config.trajectory_n - 1)

                t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen])
                agent.perceive(t_q[0])

                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)
                    replay_full_episode = replay_full_episode or e

            state = next_state

        if done:
            t_q.popleft()
            transitions = set_n_step(t_q, Config.trajectory_n)
            for t in transitions:
                agent.perceive(t)
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)
                    replay_full_episode = replay_full_episode or e

            if agent.replay_memory.full():
                scores.append(score)
                agent.sess.run(agent.update_target_net)

            # Print progress
            if replay_full_episode is not None:
                print(f"Episode: {e}, Trained Episode: {e - replay_full_episode}, Score: {score}, "
                      f"Memory Length: {len(agent.replay_memory)}, Epsilon: {agent.epsilon:.4f}")

            # Early stopping check
            if score >= score_threshold:
                early_stopping_counter += 1
                print(f"Achieved score of {score}. Early stopping counter: {early_stopping_counter}/3")

                if early_stopping_counter >= early_stopping_threshold:
                    print("Early stopping triggered: Achieved score of 500 for 3 consecutive episodes.")
                    break
            else:
                early_stopping_counter = 0  # Reset counter if the score is below the threshold

        if len(scores) >= Config.episode:
            break

        e += 1

    # Measure memory and time
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory

    

    # Save POfD scores to a file
    with open('dqfd.p', 'wb') as f:
        pickle.dump(scores, f)
    print("DQfD scores saved to dqfd.p.")

    metrics = {
        "time": elapsed_time,
        "memory_used": memory_used,
    }



    return scores, metrics



def run_POfD(index, env, demo_transitions=None):
    """
    Train the POfDV2 agent using only the existing replay buffer (demo.p).
    Parameters:
        index (int): Identifier for the training session.
        env (gym.Env): The training environment.
        demo_transitions (deque): Expert demonstration transitions (loaded from demo.p).
    Returns:
        list: Scores achieved during training.
    """
    # Load pre-existing demo data
    demo_path = Config.DEMO_DATA_PATH
    try:
        with open(demo_path, 'rb') as f:
            demo_transitions = pickle.load(f)
            demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
            assert len(demo_transitions) == Config.demo_buffer_size
            print(f"Loaded {len(demo_transitions)} demo transitions from {demo_path}.")
    except Exception as e:
        print(f"Error loading demonstrations: {e}")
        return

    # Initialize the POfDV2 agent
    with tf.compat.v1.variable_scope('POfD_' + str(index)):
        agent = POfDV2(env, Config(), demo_transitions)

    # Pre-train using only the loaded demo data
    # print("Pre-training with demo data...")
    # agent.pre_train()

    scores = []
    early_stopping_counter = 0
    early_stopping_threshold = 3
    score_threshold = 500
    e = 0

    
    # Measure memory and time
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB


    # Training loop
    while True:
        done, score, state = False, 0, env.reset()
        trajectory = []  # Store trajectory for discriminator and reward updates

        while not done:
            # Select action using policy
            #action = agent.egreedy_action(state)
            action = agent.sample_action(state)

            next_state, reward, done, _ = env.step(action)
            score += reward

            # Collect the transition in the trajectory
            trajectory.append((state, action, reward))
            state = next_state

        # Update rewards in trajectory using the discriminator (demo-only logic)
        agent.update_rewards_with_discriminator(trajectory)

        # Train the policy and discriminator
        agent.train_policy(trajectory)
        agent.train_discriminator(trajectory, demo_transitions)

        # Append score
        scores.append(score)
        print(f"Episode: {e}, Score: {score}")

        # Early stopping logic
        if score >= score_threshold:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_threshold:
                print(f"Early stopping triggered: Achieved score >= {score_threshold} for {early_stopping_threshold} consecutive episodes.")
                break
        else:
            early_stopping_counter = 0

        if len(scores) >= Config.episode:
            break

        e += 1
    # Measure memory and time
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Save scores to file
    # Save POfD scores to a file
    with open('pofd.p', 'wb') as f:
        pickle.dump(scores, f)
    print("POfD scores saved to pofd.p.")

    metrics = {
        "time": elapsed_time,
        "memory_used": memory_used,
    }


    return scores, metrics

def run_and_record_best(env, agent, num_episodes=300):
    recorder = BestEpisodesRecorder(max_episodes=5)

    for episode in range(num_episodes):
        done = False
        score = 0
        state = env.reset()
        trajectory = []

        while not done:
            action = agent.sample_action(state)  # Replace with your action logic
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            score += reward

        recorder.add_episode(score, trajectory)

        print(f"Episode: {episode}, Score: {score}")

    return recorder.get_best_episodes()


def replay_and_record(env, best_episodes):
    # Wrap the environment with Monitor to record video
    env = Monitor(env, './videos', force=True)

    for score, trajectory in best_episodes:
        state = env.reset()
        for state, action, _ in trajectory:
            env.render()
            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run POfDV2, DQfD, or populate demo data with DDQN")
    parser.add_argument("--agent", type=str, choices=["POfDV2", "DQfD", "DDQN"], required=True,
                        help="Specify the agent to run (POfDV2, DQfD, or DDQN to populate demo data)")
    parser.add_argument("--demo_path", type=str, default="demo.p", help="Path to save or load demo data (default: demo.p)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment to use (default: CartPole-v1)")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to train or populate (default: 300)")

    # Parse arguments
    args = parser.parse_args()

    # Load the specified environment
    env = gym.make(args.env)
    
    #env = Monitor(env, './videos', force=True)  # Add this line to save videos
    env = wrappers.Monitor(env, '/tmp/CartPole-v1', force=True)  # Optional for monitoring logs

    if args.agent == "DDQN":
        # Run DDQN to populate demo.p
        print("Running DDQN to populate demo.p...")
        get_demo_data(env)
        print(f"Demo data saved to {args.demo_path}.")
    elif args.agent == "DQfD":
        # Run DQfD training
        print("Running DQfD training...")
        scores = run_DQfD(0, env)
        print(f"DQfD training completed. Scores: {scores}")
    elif args.agent == "POfDV2":
        # Run POfDV2 training
        print("Running POfDV2 training...")
        scores = run_POfD(0, env)
        print(f"POfDV2 training completed. Scores: {scores}")
    else:
        print(f"Unknown agent type: {args.agent}")

    # Close the environment
    env.close()


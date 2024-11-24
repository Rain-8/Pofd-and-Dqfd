## POFD vs DQFD

We use Google Deepmind's reasearch on - "Deep Q-learning from Demonstrations" and "Policy Optimization with Demonstrations" from ICML 2018 on the cartpole environment.

DQFD Model - This research paper introduces Deep Q-learning from Demonstrations (DQfD), a deep reinforcement learning algorithm designed to overcome the significant data requirements of existing methods. DQfD leverages pre-existing demonstration data to drastically improve initial performance and accelerate learning, even with limited demonstrations. The algorithm combines temporal difference updates with supervised learning from demonstrations, significantly outperforming existing methods across various tasks. This improvement allows application to real-world scenarios lacking accurate simulators but possessing prior operational data, unlike traditional deep reinforcement learning which requires extensive training. The authors demonstrate DQfD's superiority across multiple game environments, showcasing its real-world applicability.

POFD Model - This research paper introduces Policy Optimization from Demonstration (POfD), a novel reinforcement learning method designed to overcome the challenge of exploration in sparse-reward environments. Unlike existing methods that struggle with limited or imperfect demonstrations, POfD leverages available demonstrations to guide exploration by matching the learned policy's occupancy measure to that of the demonstrations. This approach implicitly shapes the reward function, leading to provably improved policy optimization and state-of-the-art performance on benchmark tasks. The authors achieve this through a theoretically-grounded method that combines elements of reward shaping and generative adversarial training, resulting in an algorithm that is both effective and compatible with various policy gradient methods

The game we have used for our implementation is the cartpole- 

CartPole-v1 is one of OpenAI’s environments that are open source. The “cartpole” agent is a reverse pendulum where the “cart” is trying to balance the “pole” vertically, with a little shift of the angle.

The only forces that can be applied are +1 and -1, which translates to a movement of either left or right. If the cart moves more than 2.4 units from the center the episode is over. If the angle is moved more than 15 degrees from the vertical the episode is over. The reward is +1 for every timestamp that the episode is not over.

![figure_0](/images/cartpole.png)

We have structured our model as follows- 
```
    DQFD/
    ├── __pycache__/              # Compiled Python files (automatically generated, not tracked in Git)
    ├── images/                   # Directory for plots or images used for analysis
    ├── model/                    # Directory for saved models
    ├── Config.py                 # Configuration file for environment and hyperparameters
    ├── ddqn_mean_scores.p        # Pickle file containing average scores from DDQN runs
    ├── demo.p                    # Pickle file containing expert demonstration data
    ├── CartPole.py               # Main script for running POfD, DQfD, and DDQN training
    ├── DQfD_V3.py                # Implementation of DQfD algorithm
    ├── dqfd.p                    # Pickle file storing DQfD training scores
    ├── DQfDDDQN.py               # Implementation of Double DQN used to generate demo data
    ├── Memory.py                 # Memory buffer class for experience replay
    ├── plot_ddqn_dqfd.py         # Script for visualizing and comparing results
    ├── pofd_scores.p             # Pickle file storing POfD training scores
    ├── pofd_vs_dqfd.py           # Script for comparison between POfD and DQfD
    ├── pofd.p                    # Pickle file storing POfD-specific data
    ├── pofdv2.py                 # Implementation of the POfD algorithm
    └── README.md                 # Documentation describing the repository and how to use it
```
We try to obtain the results demonstrated in POFD for cartpole environment.

For DQFD - we refer to existing code base for algorithm - https://github.com/go2sea/DQfD/tree/master

For POFD, we did not find any existing implementation, so we implemented our own algorithm using the model presented in the paper.

Algorithm from paper:

![figure_1](/images/pofd_algorithm.png)

Our implementation:

The architecture described in the POfDV2 class aligns with the provided algorithm (Policy Optimization with Demonstrations). Here's a breakdown of how this class and the methods correspond to the algorithm and how they work:

The class implements the following main parts of the algorithm:

Inputs:
Expert Demonstrations (demo_transitions): These are pre-collected expert trajectories passed into the class.
Environment (env): The environment in which the agent operates, e.g., CartPole.
Policy and Discriminator Networks: These are two neural networks:
Policy Network (policy_network): For selecting actions.
Discriminator Network (discriminator_network): To distinguish between expert and agent-generated transitions.

Initialization:

```
    self._build_networks()
    self.sess = tf.compat.v1.Session()
    self.sess.run(tf.compat.v1.global_variables_initializer())
    self.pre_train()
```

Input Parameters: The class accepts expert demonstrations and initializes the policy and discriminator parameters.
Pretraining: The policy is pretrained using expert demonstrations. This corresponds to the initialization phase of the algorithm.

Policy Network:

```
    with tf.compat.v1.variable_scope("policy_network"):
        hidden = tf.compat.v1.layers.dense(self.state_input, 128, activation=tf.nn.relu)
        self.policy_logits = tf.compat.v1.layers.dense(hidden, self.env.action_space.n, activation=None)
        self.policy_probs = tf.nn.softmax(self.policy_logits, axis=-1)

```

The policy network (πθ) is designed to map states to action probabilities.
Loss Function:
Surrogate Loss: tf.minimum(surrogate1, surrogate2) implements the clipped PPO loss.
Entropy Regularization: - self.lambda2 * tf.reduce_mean(entropy) encourages exploration.

Discriminator Network:

```
    with tf.compat.v1.variable_scope("discriminator_network"):
        disc_hidden = tf.compat.v1.layers.dense(self.disc_input, 128, activation=tf.nn.relu)
        self.disc_output = tf.compat.v1.layers.dense(disc_hidden, 1, activation=tf.nn.sigmoid)

```
The discriminator (Dw) distinguishes between agent-generated (τ) and expert (DE) trajectories.

Loss Function:

```
    self.disc_loss = -tf.reduce_mean(
        self.disc_labels * tf.math.log(self.disc_output + 1e-8) +
        (1 - self.disc_labels) * tf.math.log(1 - self.disc_output + 1e-8)
    )

```
This aligns with the gradient update step for the discriminator in the algorithm.

Sampling and Updating Rewards:

1. Sampling Trajectories (Di): Actions are sampled using the current policy.

```
    trajectory.append((state, action, reward))
    action = agent.sample_action(state)

```

2. Updating Rewards with Discriminator: Rewards are adjusted based on the discriminator output (Dw(s, a)).

```
    updated_reward = reward - self.lambda1 * np.log(disc_value + 1e-8)

```
Policy Gradient Update:

```
    self.sess.run(self.policy_optimizer, feed_dict={
        self.state_input: states,
        self.action_input: actions,
        self.advantage_input: discounted_rewards,
        self.old_log_probs: old_log_probs
    })

```
This step updates the policy using the gradient described in the algorithm:
Ê_{D_i}[∇_θ log π_θ(a|s)Q'(s,a)] - λ_2 ∇_θ H(π_{θ_i})
Here:
- Q'(s,a) is represented by `discounted_rewards`.
- The first term adjusts the policy based on action-state probabilities and discounted rewards.
- The second term regularizes the policy entropy, encouraging exploration.


Overall Workflow

1. Pretrain the Policy:
- Use expert demonstrations (D_E) to initialize the policy network.
2. Training Loop:
   - Sample Trajectories: Generate trajectories (D_i) using the current policy.
   - Update the Discriminator: Train the discriminator using expert and agent trajectories.
   - Update Rewards: Modify rewards based on the discriminator output.
   - Policy Update: Use PPO to optimize the policy.

Alignment with the Algorithm

1. Expert Demonstrations (D_E):
- Passed as `demo_transitions` during initialization.
2. Policy Update:
- Implements PPO with entropy regularization (λ_2).
- Adjusts policy using discriminator-based reward modification (λ_1).
3. Discriminator Update:
- Distinguishes between expert and agent trajectories.
- Provides improved rewards for training the policy.


Paper results for Cartpole:

1. Generating Demonstrations
Single Imperfect Trajectory:
• Only one imperfect trajectory is used as a demonstration for POfD. This trajectory is not optimal or fully trained, simulating a realistic scenario where perfect expert demonstrations may not be available.
• Using a single trajectory tests the ability of POfD to learn effectively from limited and imperfect data.
How the Demonstration is Generated:
• The agent is trained insufficiently using TRPO (Trust Region Policy Optimization) in a dense environment (where rewards are abundant and provide more feedback).
• After this partial training, a random trajectory is sampled from the agent’s behavior.
• This trajectory becomes the demonstration used for training POfD.

2. Evaluation Metrics
The effectiveness of POfD is evaluated using two metrics:
(a) Training Curves
• The method is run 5 times, each with a different random initialization.
• Training curves are analyzed to evaluate how well POfD facilitates exploration in sparse environments.
• Exploration is critical in sparse environments because rewards are sparse, and the agent must intelligently explore to find the target states.
(b) Empirical Returns
• After training, the learned policy is evaluated by calculating empirical returns:
  - Run the policy for 500 episodes (or trials).
  - Record the cumulated rewards over these trials.
  - Compute the average return to quantify the policy's overall performance.

Their Result for cartpole:

![figure_2](/images/paper_res.png)


Our Implementation and Results:

First we run the get_demo_data() function which uses DDQN for generating the demo data and storing it in demo.p pickle file

```
    python CartPole.py --agent DDQN
```

This will fill the demo memory which will act as expert for the DQFD and POFD training


![figure_3](/images/demo_load.png)

Then we pre-train the DQFD agent by loading demo.p into it and then we see the training results and store the scores in dqfd.p pickle file

```
    python Cartpole.py --agent DQfD --demo_path demo.p  
```

Then we do the same for POFD agent and store the scores in pofd.p pickle file
 
```
    python Cartpole.py --agent POfDV2 --demo_path demo.p
```

Then we plot the scores vs number of episodes for both DQFD (red) and POFD (green)
```
    python pofd_vs_dqfd.py
```
Below are the results attached for pre-train steps=1400, Learning rate= 0.0008 

Config -

![figure_4](/images/config_0008.png)

DQFD -

![figure_5](/images/dqfd_0008.png)

POFD -

![figure_6](/images/pofd_0008.png)

Results -

![figure_7](/images/res_0008.png)








# DQfD

An implementation of DQfD（Deep Q-learning from Demonstrations) raised by DeepMind:Learning from Demonstrations for Real World Reinforcement Learning

It also compared DQfD with Double DQN on CartPole game, and the comparison shows that DQfD outperforms Double DQN.

Algorithm is different between Version3(DQfD.py) and Version-1(DQfDDDQN.py). Compared to V1, the V3 DQfD added:
 
```
    prioritized replay
    n-step TD loss
    importance sampling
```


## Comparison between DQfD(V3) and Double DQN

Compared to V1, the V3 DQfD added prioritized replay, n-step TD loss, importance sampling.

![figure_0](/images/dqfd-v3_vs_ddqn.png)

Note: In my experiments on CartPole, the n-step TD loss is Counterproductive and leads to worse performance. And the parameter λ for loss_n_step for the result you see above is 0. Maybe I implement n-step TD loss in wrong way and I hope someone can explain that.

## Comparison between DQfD(V1) and Double DQN

![figure_1](/images/figure_1.png)
![figure_2](/images/figure_2.png)
![figure_3](/images/figure_3.png)

At the end of training, the epsilon used in greedy_action is 0.01.


## Get the expert demo transitions

Compared to double DQN, a improvement of DQfD is pre-training. DQfD initially trains solely on the demonstration data before starting any interaction with the environment. This code used a network fine trained by Double DQN to generate the demo data.

You can see the details in function:
```
  get_demo_data()
```

## Get Double DQN scores

For Comparison, I first trained an network through Double DQN, witch has the same net with the DQfD.
```
    # --------------------------  get DDQN scores ----------------------------------
    ddqn_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DDQN(i, env)
        ddqn_sum_scores = np.array([a + b for a, b in zip(scores, ddqn_sum_scores)])
    ddqn_mean_scores = ddqn_sum_scores / Config.iteration
```

## Get DQfD scores

```
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
```
## Map

Finally, we can use this function to show the difference between Double DQN and DQfD.
```
    map(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores, xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
```




##POFD vs DQFD

We use Google Deepmind's reasearch on - "Deep Q-learning from Demonstrations" and "Policy Optimization with Demonstrations" from ICML 2018 on the cartpole environment.

DQFD Model - This research paper introduces Deep Q-learning from Demonstrations (DQfD), a deep reinforcement learning algorithm designed to overcome the significant data requirements of existing methods. DQfD leverages pre-existing demonstration data to drastically improve initial performance and accelerate learning, even with limited demonstrations. The algorithm combines temporal difference updates with supervised learning from demonstrations, significantly outperforming existing methods across various tasks. This improvement allows application to real-world scenarios lacking accurate simulators but possessing prior operational data, unlike traditional deep reinforcement learning which requires extensive training. The authors demonstrate DQfD's superiority across multiple game environments, showcasing its real-world applicability.

POFD Model - This research paper introduces Policy Optimization from Demonstration (POfD), a novel reinforcement learning method designed to overcome the challenge of exploration in sparse-reward environments. Unlike existing methods that struggle with limited or imperfect demonstrations, POfD leverages available demonstrations to guide exploration by matching the learned policy's occupancy measure to that of the demonstrations. This approach implicitly shapes the reward function, leading to provably improved policy optimization and state-of-the-art performance on benchmark tasks. The authors achieve this through a theoretically-grounded method that combines elements of reward shaping and generative adversarial training, resulting in an algorithm that is both effective and compatible with various policy gradient methods

The game we have used for our implementation is the cartpole- 

CartPole-v1 is one of OpenAI’s environments that are open source. The “cartpole” agent is a reverse pendulum where the “cart” is trying to balance the “pole” vertically, with a little shift of the angle.

The only forces that can be applied are +1 and -1, which translates to a movement of either left or right. If the cart moves more than 2.4 units from the center the episode is over. If the angle is moved more than 15 degrees from the vertical the episode is over. The reward is +1 for every timestamp that the episode is not over.

![figure_0](C:\Users\SowmyaG\projects\cartpole\DQfD\images\cartpole.png)

We have structured our model as follows- 
```
    DQFD/
    ├── __pycache__/              # Compiled Python files (automatically generated, not tracked in Git)
    ├── images/                   # Directory for plots or images used for analysis
    ├── model/                    # Directory for saved models
    ├── Config.py                 # Configuration file for environment and hyperparameters
    ├── ddqn_mean_scores.p        # Pickle file containing average scores from DDQN runs
    ├── demo.p                    # Pickle file containing expert demonstration data
    ├── DQfD_CartPole.py          # Main script for running POfD, DQfD, and DDQN training
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


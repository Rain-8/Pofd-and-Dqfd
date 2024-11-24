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

First we run the get_demo_data() function which uses DDQN for generating the demo data and storing it in demo.p pickle file

```
    python CartPole.py --agent DDQN
```

This will fill the demo memory which will act as expert for the DQFD and POFD training


![figure_1](/images/demo_load.png)

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

![figure_2](/images/config_0008.png)

DQFD -

![figure_3](/images/dqfd_0008.png)

POFD -

![figure_4](/images/pofd_0008.png)

Results -

![figure_5](/images/res_0008.png)






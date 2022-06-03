# Udacity - Deep Reinforcement Learning - Project 2

##Project Description

In this exercise the task is to create and train an agent to control an arm with two joint in a three-dimensional space.
Each of the joints expects continuous commands in the range -1 to +1.
The agent shall control the arm in such a way that the end of the arm (the "hand") is in a defined area.
The defined are has the shape of a sphere and is rotating around the stable side of the arm (the "shoulder").

If the hand is located within the sphere the agent is provided a reward.
The exercise description states a reward of 0.1 but the measured maximum reward was approximately 0.04 [1](PROBLEM_SOURCE_1), [2](PROBLEM_SOURCE_2).
This difference makes the exercise more difficult to solve and one episode has to have at least 770 steps for a perfect agent.

The environment vectors have the following dimensions:
- State vector size: 33
- Action vector size: 4

For this exercise different environments are provided:
- an environment with one arm (`One`)
- an environment with twenty arms (`Twenty`)

In case of the world with multiple arms both the state vector and the action vector both are an array providing the state and excepting the action for each agent.

The environment is considered as solved if the agent(s) reach an average score of 30 or higher over 100 episodes.

The machine leaning approach for this exercise is free to choose, therefore I decided to use [PPO](PPO) since it was the most appealing algorithm for me.

[PPO]: https://openai.com/blog/openai-baselines-ppo/
[PROBLEM_SOURCE_1]: https://knowledge.udacity.com/questions/32300
[PROBLEM_SOURCE_2]: https://knowledge.udacity.com/questions/558456

## Solution Description

The environment is solved with the code and the agent in this repository.
The solution is build in various python files.
How to install the program and the dependencies is described in the `README.md` file in the section *Dependencies* and how operate the program is described in the `README.md` file in the section *Execution*.

In order to solve the environment the agent utilizes the PPO algorithm.
According to the PPO algorithm the agent contains an actor and a critic.
In order to train the agent trajectories of the arm(s) have to be collected in the environment.
In order to allow constant learning the actions taken by the agent include some factor of randomness.

### General Architecture

Generally PPO consists the actor and the critic.
Both are neuronal networks with various layers and activation functions.
The network for the actor and the critic can share some common layers, but do not have to.
In the implementation of PPO in this repository the agent and actor network are completely separate.
The actor network is responsible for action selection.
The critic network is responsible for action assessment and therefor highly responsible for effective training.

#### Neuronal Network
The neuronal network is a group of layers of neurons and connections between the neurons of one layer to the neurons of the next layer.
The input layer is the entry point of information and the output layer is the exit point of the network where a decision is made.
Information that is passed into the network activates the neurons in the input layer which again activates the next layer and so on until the output layer is activated and a decision was made.
The intensity of the activation depends on the weights stored in the connection of the neurons.
During the training of the agent these weights are adopted to achieve the desired activation in the output layer corresponding to the input at the input layer.

### Specific Architecture

The implementation in this repository is based on the tutorial [Coding PPO from Scratch](PPO_FROM_SCRATCH) and the according GitHun [repository](PPO_FROM_SCRATCH_GITHUB).
Nevertheless, this tutorial was inspiration and guide and the code in this repository is written for this exercise.

[PPO_FROM_SCRATCH]: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
[PPO_FROM_SCRATCH_GITHUB]: https://github.com/ericyangyu/PPO-for-Beginners

The solution for the environment introduced above is split up into various files:

- `continuous.py`:
  The entry point for the code execution.
  The main logic of the execution modes (`tune`,`train` and `show` - for more details please have a look at the README.md file) is in this file.
- `src/agent.py`:
  In this file the agent is defined.
  This includes the creation of the actor and critic network aas well as the functions to train these networks.
- `src/environment.py`:
  In this file the environment API is reduced to those features which are necessary for this solution.
  The functions `state_size()` and `action_size()` allow swapping between the environment with a single arm ort with multiple arms without any other changes in the code (still changes in the hyperparameter might be required to efficiently solve the environment).
  The function `number_of_agents()` allows to get information about the number of arms in the environment.
- `src/network.py`:
  In this file the neuronal network which can be applied for the actor and the critic is defined.
  The structure of the network can be defined in the hyperparameters.
  This includes the number of layers, the size of each layer and the activation function which the layer output are applied to.
  Additionally, a final optional output function can be applied to the neuronal network.
- `src/utils.py`:
  This file contains functionalities which is neither part of the agent nor the environment.
- `src/hyperparameters/hyperparameters.py`:
  File holding the hyperparameters for the training function.
  For details please have a look at the according file.
- `src/hyperparameters/hyperparameters_range.py`:
  File holding the hyperparameter ranges for the tuning function.
  For details please have a look at the according file.

The agent has a dedicated neuronal network for the actor and for the critic.
Their layer structure and activation functions are defined in the hyperparameters.
Both networks use gradient ascend for the training process.
The loss functions are defined according to the PPO algorithm.

The agent shown in this repository learns trajectories.
Each step of the trajectories can be utilized for training multiple times.
After a trajectory has been learned a new trajectory with the updated networks can be collected.

### Findings

While solving this exercise various configurations have been tried.

The aim for me was to solve the environment with an episode length of 1000 steps.
This was possible with the hyperparameters shown below.

The following parameters can be part of the hyperparameter tuning process:
- `episodes`: number of episodes
- `trajectories`: number of trajectories for each episode
- `steps`: number of steps for each episode (note not the number of steps for a trajectory)
- `gamma`: discount factor for future rewards
- `clip`: clipping value for the PPO algorithm
- `training_iterations`: training iterations for each step of a trajectory

- `actor_layers`: network architecture for the actor network
- `actor_activation_function`: activation function applied after each layer of the neuronal network of the actor (except the last)
- `actor_output_function`: activation function after the last layer of the neuronal network of the actor, if set to `None` no such function is applied
- `actor_optimizer`: optimizer function of the actor
- `actor_optimizer_learning_rate`: optimizer learning rate of the actor
- `actor_covariance`: size of the covariance of the actor action selection

- `critic_layers`: network architecture for the critic network
- `critic_activation_function`: activation function applied after each layer of the neuronal network of the critic (except the last)
- `critic_output_function`: activation function after the last layer of the neuronal network of the critic, if set to `None` no such function is applied
- `critic_optimizer`: optimizer function of the critic
- `critic_optimizer_learning_rate`: optimizer learning rate of the critic

The following hyperparameter work very well for the `Twenty` environment:
```python
hp["episodes"] = 1000
hp["trajectories"] = 2
hp["steps"] = 1000

hp["gamma"] = 0.90
hp["clip"] = 0.200

hp["training_iterations"] = 10

hp["actor_layers"] = [128, 64, 32]
hp["actor_activation_function"] = torch.nn.ReLU
hp["actor_output_function"] = None
hp["actor_optimizer"] = torch.optim.Adam
hp["actor_optimizer_learning_rate"] = 0.005
hp["actor_covariance"] = 1.0

hp["critic_layers"] = [128, 64, 32]
hp["critic_activation_function"] = torch.nn.ReLU
hp["critic_output_function"] = None
hp["critic_optimizer"] = torch.optim.Adam
hp["critic_optimizer_learning_rate"] = 0.005
```

With these values it was possible to solve the environment after 150 to 170 episodes.
The score graph created with these parameters is shown in the file `training.png`.

The agent is also able to solve the `One` environment, but for this environment the parameters have not been tweaked.

### Improvements

In the task description a benchmark solving the environment in about 180 episodes is provided.
The solution presented in this repository is capable to solve the environment in about 150 to 170 episodes with the parameters provided above.

Since the maximum score for the environment with 1000 steps per episode is approximately 39 and the environment is solved with an average score of 30over 100 episodes the solution provided in this repository is close to a perfect solution.
Still there is some room for improvements.

The following changes could improve the learning process and resulting neuronal network:
- Extended Hyperparameter Tuning:
  An exhaustive grid search for the best hyperparameters could improve the learning process and the resulting neuronal network.
- Changing the Network Architecture:
  The training process might be improved by a some shared layers between the actor network and the critic network.
- Improving Execution Performance:
  The code written to solve the environment was not written with efficiency in mind.
  Therefore, the code potentially has inefficiencies.
  Optimizing the code might lead to significant boost in efficiency.

## Summary

The exercise was very hard but interesting to solve.
Unfortunately the code provided in the Pong workspace in the udacity course is not very well documented and therefore not ideal to study the functionality of PPO.
With the tutorial mentioned above it was possible to get a better understanding of PPO and therefore to solve the exercise.

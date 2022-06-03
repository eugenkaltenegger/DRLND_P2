# Udacity - Deep Reinforcement Learning - Project 1

## Project Description

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

### Dependencies

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

In order to operate this repository it is necessary to download the executables for the environment and to create a conda environment.
For the world with a single arm the executable can be downloaded [here](One) and for the world with multiple arms the executable can be downloaded [here](Twenty).

[One]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
[Twenty]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip

The downloaded `.zip` files have to be extracted and placed in the folder `env`.
The resulting file structure should look like shown below:
```bash
├── env
│    ├── Reacher_Linux_One
│    │   └── Reacher_Linux
│    │       ├── Reacher_Data
│    │       ├── Reacher.x86
│    │       └── Reacher.x86_64
│    └── Reacher_Linux_Twenty
│        └── Reacher_Linux
│            ├── Reacher_Data
│            ├── Reacher.x86
│            └── Reacher.x86_64
```

To create a conda environment and install the packages required for this repository run the following command:
```bash
conda env create --file requirements.yaml
```

This conda environment has to be activated with the following command:
```bash
conda activate kalteneger_p2_continuous-control
```

With the active conda environment and the installed dependencies the preparation to run the code is completed.

### Execution

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

To execute the code run the following command in a terminal with the active conda environment:
```bash
python3 continuous-control.py <mode> <world>
```

To code provided in this repository has three operation modes:
- `tune`: hyperparameter tuning, the list of hyperparameters set in form of lists in the file `hyperparameters_range.py` in the ordered dict is applied.
  The results of each hyperparameter combination are shown and finally the combination solving the environment after the least steps with the highest score is listed.
  The graph showing the scores of the best training with the best hyperparameter set is stored in `tuning.png`.
- `train`: training the agent, the agent is trained with the hyperparameters set in the file `hyperparameters.py` in the ordered dict.
  The graph for the score and the agent state are stored in `training.png` and `continuous-control.pth`.
- `show`: showing the operation of the trained agent.
  The simulation is started with visualization and the trained agent is operating in the environment.
  This mode is for visualization purposes only.

The solution can either operate on the `One` environment or the `Twenty` environment.
The world argument has to be either:
- `One`
- `Twenty` (default)

To start the program the command could look like:
```bash
python3 continuous-control.py show Twenty
```

**Note:** the provided agent snapshot (`continuous-control.pth`) was created for the `Twenty` environment.
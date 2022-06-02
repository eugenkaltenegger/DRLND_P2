#!/usr/bin/env python3

import torch

from collections import OrderedDict


class Hyperparameters:
    """
    class to hold hyperparameters
    """

    def __init__(self):
        """
        constructor for hyperparameters class
        """
        hp = OrderedDict()

        hp["episodes"] = 250
        # trajectories per episode
        hp["trajectories"] = 4
        # steps per episode
        hp["steps"] = 250

        hp["gamma"] = 0.90
        hp["clip"] = 0.200

        # iterations the agent is learning one step
        hp["training_iterations"] = 5

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

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp

#!/usr/bin/env python3

import torch

from collections import OrderedDict


class HyperparametersRange:
    """
    class to hold hyperparameter ranges
    """

    def __init__(self):
        """
        constructor for hyperparameter ranges class
        """
        hp = OrderedDict()

        # maximum episodes
        hp["episodes"] = [250]
        # trajectories per episode
        hp["trajectories"] = [1, 2, 4]
        # steps per episode
        hp["steps"] = [1000]

        hp["gamma"] = [0.99]
        hp["clip"] = [0.20]

        # iterations the agent is learning one step
        hp["training_iterations"] = 5

        hp["actor_layers"] = [[128, 64, 32]]
        hp["actor_activation_function"] = [torch.nn.ReLU]
        hp["actor_output_function"] = [None]
        hp["actor_optimizer"] = [torch.optim.Adam]
        hp["actor_optimizer_learning_rate"] = [0.005]

        hp["critic_layers"] = [[128, 64, 32]]
        hp["critic_activation_function"] = [torch.nn.ReLU]
        hp["critic_output_function"] = [None]
        hp["critic_optimizer"] = [torch.optim.Adam]
        hp["critic_optimizer_learning_rate"] = [0.005]

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameter ranges as an OrderedDict
        :return: hyperparameter ranges in an OrderedDict
        """
        return self.hp

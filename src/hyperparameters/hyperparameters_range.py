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

        hp["episodes"] = [1000]
        hp["rollout_steps"] = [60]

        hp["gamma"] = [0.99]
        hp["decay"] = [0.99]
        hp["clip"] = [0.20]
        hp["kl_difference_limit"] = [0.02]

        hp["actor_layers"] = [[128, 64, 32]]
        hp["actor_activation_function"] = [torch.nn.ReLU]
        hp["actor_output_function"] = [torch.nn.Tanh]
        hp["actor_training_iterations"] = [10]
        hp["actor_optimizer"] = [torch.optim.Adam]
        hp["actor_optimizer_learning_rate"] = [0.005]

        hp["critic_layers"] = [[128, 64, 32]]
        hp["critic_activation_function"] = [torch.nn.ReLU]
        hp["critic_output_function"] = [torch.nn.ReLU]
        hp["actor_training_iterations"] = [10]
        hp["critic_optimizer"] = [torch.optim.Adam]
        hp["critic_optimizer_learning_rate"] = [0.005]

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameter ranges as an OrderedDict
        :return: hyperparameter ranges in an OrderedDict
        """
        return self.hp

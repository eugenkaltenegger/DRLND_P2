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

        hp["episodes"] = 1000
        hp["rollouts"] = 10
        hp["steps"] = 10

        hp["trajectories_per_episode"] = 10
        hp["steps_per_trajectory"] = 100

        hp["gamma"] = 0.99
        hp["decay"] = 0.99
        hp["clip"] = 0.200
        hp["kl_difference_limit"] = 0.200

        hp["actor_layers"] = [64, 32, 16]
        hp["actor_activation_function"] = torch.nn.Tanh
        hp["actor_output_function"] = None
        hp["actor_training_iterations"] = 1
        hp["actor_optimizer"] = torch.optim.Adam
        hp["actor_optimizer_learning_rate"] = 0.005
        hp["actor_covariance"] = 1.0

        hp["critic_layers"] = [64, 32, 16]
        hp["critic_activation_function"] = torch.nn.ReLU
        hp["critic_output_function"] = None
        hp["critic_training_iterations"] = 1
        hp["critic_optimizer"] = torch.optim.Adam
        hp["critic_optimizer_learning_rate"] = 0.005

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp

    @staticmethod
    def pass_trough(any_value):
        return any_value

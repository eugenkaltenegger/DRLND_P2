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
        hp_dict = OrderedDict()

        hp_dict["episodes"] = 1000
        hp_dict["steps_per_episode"] = 100

        hp_dict["gamma"] = 0.995
        hp_dict["clip"] = 0.200

        hp_dict["actor_layers"] = [128, 64, 32]
        hp_dict["actor_activation_function"] = torch.nn.ReLU
        hp_dict["actor_output_function"] = torch.nn.Tanh
        hp_dict["actor_optimizer"] = torch.optim.Adam
        hp_dict["actor_optimizer_learning_rate"] = 0.05

        hp_dict["critic_layers"] = [128, 64, 32]
        hp_dict["critic_activation_function"] = torch.nn.ReLU
        hp_dict["critic_output_function"] = torch.nn.Tanh
        hp_dict["critic_optimizer"] = torch.optim.Adam
        hp_dict["critic_optimizer_learning_rate"] = 0.05

        self.hp_dict = hp_dict

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp_dict

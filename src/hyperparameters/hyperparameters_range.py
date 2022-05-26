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
        hpr_dict = OrderedDict()
        hpr_dict["episodes"] = [7]
        hpr_dict["steps_per_episode"] = [3]

        hpr_dict["gamma"] = [0.995]
        hpr_dict["clip"] = [0.200]

        hpr_dict["actor_layers"] = [[128, 64, 32]]
        hpr_dict["actor_activation_function"] = [torch.nn.ReLU]
        hpr_dict["actor_output_function"] = [torch.nn.Tanh]
        hpr_dict["actor_optimizer"] = [torch.optim.Adam]
        hpr_dict["actor_optimizer_learning_rate"] = [0.005]

        hpr_dict["critic_layers"] = [[128, 64, 32]]
        hpr_dict["critic_activation_function"] = [torch.nn.ReLU]
        hpr_dict["critic_output_function"] = [torch.nn.Tanh]
        hpr_dict["critic_optimizer"] = [torch.optim.Adam]
        hpr_dict["critic_optimizer_learning_rate"] = [0.005]

        self.hpr_dict = hpr_dict

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameter ranges as an OrderedDict
        :return: hyperparameter ranges in an OrderedDict
        """
        return self.hpr_dict

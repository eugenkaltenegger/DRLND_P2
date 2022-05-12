#!/usr/bin/env python3
import sys
import torch

from src.environment import Environment
from src.policy import Policy

import torch.nn as nn


class ContinuousControl:

    def __init__(self):
        # device variable
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # environment variables
        self._environment_name = None
        self._environment = None

        # score variables
        self._absolute_scores = None
        self._average_scores = None

    def setup(self, mode=None, environment_name=None):
        mode = mode if mode is not None else "show"
        self._environment_name = environment_name if environment_name is not None else "Twenty"

        if mode == "train":
            self.train()
            self.plot()

        if mode == "tune":
            self.tune()
            self.plot()

        if mode == "show":
            self.show()

    def reset_environment(self, environment_name, enable_graphics, train_environment):
        """
        function to reset the environment and get an info about the environment state
            if there is no environment a suitable environment is built
        :param environment_name: ame of the environment to use, either "One" (one arm) or "Twenty" (twenty arms)
        :param enable_graphics: parameter to set whether the environment is visualized or not visualized
        :param train_environment: parameter to set whether the environment is for training or for evaluation
        :return: info about the environment state
        """
        if self._environment is None:
            self._environment = Environment(environment_name=environment_name, enable_graphics=enable_graphics)

        return self._environment.reset(train_environment=train_environment)

    def reset_agent(self):
        # TODO
        pass

    def train(self):
        # setup environment
        enable_graphics = False
        train_environment = True
        self.reset_environment(environment_name=self._environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)

        # TODO
        pass

    def tune(self):
        # setup environment
        enable_graphics = False
        train_environment = True
        self.reset_environment(environment_name=self._environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)
        # TODO
        pass

    def show(self):
        # setup environment
        enable_graphics = True
        train_environment = False
        self.reset_environment(environment_name=self._environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)
        # TODO
        pass

    def plot(self):
        # TODO
        pass


if __name__ == "__main__":
    mode_arg = sys.argv[1]
    environment_name_arg = sys.argv[2]
    ContinuousControl().setup(mode=mode_arg, environment_name=environment_name_arg)

#!/usr/bin/env python3
import sys

from src.environment import Environment


class ContinuousControl:

    def __init__(self):
        self._environment = None
        self._absolute_scores = None
        self._average_scores = None

    def setup(self, mode=None, environment_name=None):
        mode = mode if mode is not None else "show"
        environment_name = environment_name if environment_name is not None else "Twenty"

        if mode == "train":
            self.train()
            self.plot()

        if mode == "tune":
            self.tune()
            self.plot()

        if mode == "show":
            self.show()

    def reset_environment(self, environment_name, enable_graphics, train_mode):
        if self._environment is None:
            self._environment = Environment(environment_name=environment_name, enable_graphics=enable_graphics)

        return self._environment.reset(train_mode=train_mode)

    def reset_agent(self):
        # TODO
        pass

    def train(self):
        # TODO
        pass

    def tune(self):
        # TODO
        pass

    def plot(self):
        # TODO
        pass

    def show(self):
        # TODO
        pass


if __name__ == "__main__":
    mode_arg = sys.argv[1]
    environment_name_arg = sys.argv[2]
    ContinuousControl().setup(mode=mode_arg, environment_name=environment_name_arg)
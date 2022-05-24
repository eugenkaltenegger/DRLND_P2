#!/usr/bin/env python3

from collections import OrderedDict


class HyperparametersRange:
    def __init__(self):
        hp_dict = OrderedDict()
        # e.g. hp_dict["a_hyperparameter"] = [1, 2, 4]
        self.hp_dict = hp_dict

    def get_dict(self) -> OrderedDict:
        return self.hp_dict

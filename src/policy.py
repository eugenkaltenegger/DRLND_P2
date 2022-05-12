from collections import OrderedDict
from itertools import cycle

import torch.nn as nn


class Policy(nn.Module):
    """
    class to hold a policy for an agent
    """

    def __init__(self, state_size, action_size, layers, output_function):
        """
        constrictor for Policy class
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param output_function: function to process output
        """
        super(Policy, self).__init__()

        self._output_function = output_function

        input_size = state_size
        output_size = action_size

        layer_names = ["fc{}".format(counter) for counter in range(0, len(layers) + 1)]
        layer_sizes = []
        layer_sizes += [(input_size, layers[0])]
        layer_sizes += [(layers[i-1], layers[i]) for i in range(1, len(layers))]
        layer_sizes += [(layers[-1], output_size)]
        layer_objects = [nn.Linear(layer_size[0], layer_size[1]) for layer_size in layer_sizes]
        layers_dict = OrderedDict(zip(layer_names, layer_objects))

        relu_names = ["relu{}".format(counter) for counter in range(0, len(layers))]
        relu_objects = [nn.ReLU() for _ in range(0, len(layers))]
        relu_dict = OrderedDict(zip(relu_names, relu_objects))

        key_iterators = [iter(layers_dict.keys()), iter(relu_dict.keys())]
        values_iterators = [iter(layers_dict.values()), iter(relu_dict.values())]

        key_list = list(iterator.__next__() for iterator in cycle(key_iterators))
        value_list = list(iterator.__next__() for iterator in cycle(values_iterators))

        model_dict = OrderedDict(zip(key_list, value_list))

        self.model_sequential = nn.Sequential(model_dict)

    def forward(self, state):
        """
        function to process a state
        :param state: state to process
        :return: action to take
        """
        state = self.model_sequential(state)
        return self._output_function(state)

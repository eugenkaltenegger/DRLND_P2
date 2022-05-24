#!/usr/bin/env python3

from collections import OrderedDict
from itertools import cycle

import torch


class Model(torch.nn.Module):
    """
    class to hold a policy for an agent
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 layers: [int],
                 activation_function: torch.nn.modules.Module,
                 output_function: torch.nn.modules.Module):
        """
        constructor for Policy class
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        """
        super(Model, self).__init__()

        self._setup(state_size=state_size,
                    action_size=action_size,
                    layers=layers,
                    activation_function=activation_function,
                    output_function=output_function)

    def forward(self, state: torch.Tensor):
        """
        function to process a state
        :param state: state to process
        :return: action to take
        """
        state = self._model_sequential(state)
        return self._output_function(state)

    def training_mode(self):
        """
        function to set model to training mode
        :return: nothing
        """
        self.train(True)

    def evaluation_mode(self):
        """
        function to set model to evaluation mode
        :return: nothing
        """
        self.train(False)

    def _setup(self,
               state_size: int,
               action_size: int,
               layers: [int],
               activation_function: torch.nn.modules.Module,
               output_function: torch.nn.modules.Module):
        """
        function to create a network with fully connected layers and a defined activation functions after each layer
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :return: sequential of the model dictionary
        """

        input_size = state_size
        output_size = action_size

        layer_names = ["fc{}".format(counter) for counter in range(0, len(layers) + 1)]
        layer_sizes = []
        layer_sizes += [(input_size, layers[0])]
        layer_sizes += [(layers[i - 1], layers[i]) for i in range(1, len(layers))]
        layer_sizes += [(layers[-1], output_size)]
        layer_objects = [torch.nn.Linear(layer_size[0], layer_size[1]) for layer_size in layer_sizes]
        layers_dict = OrderedDict(zip(layer_names, layer_objects))

        activation_function_names = ["af{}".format(counter) for counter in range(0, len(layers))]
        activation_function_objects = [activation_function for _ in range(0, len(layers))]
        activation_function_dict = OrderedDict(zip(activation_function_names, activation_function_objects))

        key_iterators = [iter(layers_dict.keys()), iter(activation_function_dict.keys())]
        values_iterators = [iter(layers_dict.values()), iter(activation_function_dict.values())]

        key_list = list(iterator.__next__() for iterator in cycle(key_iterators))
        value_list = list(iterator.__next__() for iterator in cycle(values_iterators))

        model_dict = OrderedDict(zip(key_list, value_list))
        model_sequential = torch.nn.Sequential(model_dict)

        self._model_sequential = model_sequential
        self._output_function = output_function

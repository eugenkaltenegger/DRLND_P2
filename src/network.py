#!/usr/bin/env python3

import torch
import typing

from collections import OrderedDict
from itertools import cycle
from torch import Tensor
from torch.nn import Module
from typing import Dict
from typing import List

# required for typehint Self
Self = typing.TypeVar("Self", bound="Network")


class Network(torch.nn.Module):
    """
    class to hold a model for an agent
    """

    def __init__(self):
        """
        initializer for model class
            this constructor requires the setup function afterwards
        """
        super(Network, self).__init__()

        self._state_size = None
        self._action_size = None
        self._layers = None
        self._activation_function = None
        self._activation_function_name = None
        self._output_function = None
        self._output_function_name = None
        self._model_sequential = None

    def forward(self, state: Tensor) -> Tensor:
        """
        function to process a state
        :param state: state to process
        :return: model output
        """
        if self._output_function is None:
            return self._model_sequential(state)
        if self._output_function is not None:
            return self._output_function(self._model_sequential(state))

    def to_checkpoint_dict(self) -> Dict:
        """
        function to export a network to the network parameters (as dict)
        :return: the network parameters (as dict)
        """
        return {"input_size": self._state_size,
                "output_size": self._action_size,
                "hidden_layers": self._layers,
                "activation_function": self._activation_function_name,
                "output_function": self._output_function_name,
                "state_dict": self.state_dict()}

    def from_checkpoint_dict(self, checkpoint: Dict) -> Self:
        """
        function to import a network from network parameters (as dict)
        :param checkpoint: the network parameters (as dict)
        :return: the network
        """
        self.setup(state_size=checkpoint["input_size"],
                   action_size=checkpoint["output_size"],
                   layers=checkpoint["hidden_layers"],
                   activation_function=checkpoint["activation_function"],
                   output_function=checkpoint["output_function"])
        self.load_state_dict(state_dict=checkpoint["state_dict"])
        return self

    def setup(self,
              state_size: int,
              action_size: int,
              layers: List[int],
              activation_function: Module,
              output_function: Module) -> Self:
        """
        function to create a network with fully connected layers and a defined activation functions after each layer
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :return: sequential of the model dictionary
        """

        self._state_size = state_size
        self._action_size = action_size
        self._layers = layers
        self._activation_function_name = activation_function
        self._activation_function = activation_function()
        self._output_function_name = output_function if output_function is not None else None
        self._output_function = output_function() if output_function is not None else None

        input_size = self._state_size
        output_size = self._action_size

        # l stands for layer
        l_names = ["fc{}".format(counter) for counter in range(0, len(self._layers) + 1)]
        l_sizes = []
        l_sizes += [(input_size, self._layers[0])]
        l_sizes += [(self._layers[i - 1], self._layers[i]) for i in range(1, len(self._layers))]
        l_sizes += [(self._layers[-1], output_size)]
        l_objects = [torch.nn.Linear(layer_size[0], layer_size[1]) for layer_size in l_sizes]
        layers_dict: Dict[str, torch.nn.Module] = OrderedDict(zip(l_names, l_objects))

        # af stands for activation function
        af_names = ["af{}".format(counter) for counter in range(0, len(self._layers))]
        af_objects = [self._activation_function for _ in range(0, len(self._layers))]
        activation_function_dict: Dict[str, torch.nn.Module] = OrderedDict(zip(af_names, af_objects))

        key_iterators = [iter(layers_dict.keys()), iter(activation_function_dict.keys())]
        values_iterators = [iter(layers_dict.values()), iter(activation_function_dict.values())]

        key_list = list(iterator.__next__() for iterator in cycle(key_iterators))
        value_list = list(iterator.__next__() for iterator in cycle(values_iterators))

        model_dict: OrderedDict[str, torch.nn.Module] = OrderedDict(zip(key_list, value_list))
        model_sequential = torch.nn.Sequential(model_dict)

        self._model_sequential = model_sequential

        return self

#!/usr/bin/env python3

import logging
import os
import torch
import unityagents

from torch import Tensor as Tensor
from typing import NoReturn
from typing import Tuple
from unityagents import UnityEnvironment


class Environment:
    """
    class to wrap the unity environment given for this exercise
    """

    def __init__(self, environment_name: str, enable_graphics: bool = False) -> None:
        """
        initializer for the Environment class
        :param environment_name: name of the environment to use, either "One" (one arm) or "Twenty" (twenty arms)
        :param enable_graphics: parameter to set whether the environment is visualized or not visualized
        """
        if environment_name is None:
            logging.error("\rMISSING ENVIRONMENT NAME ARGUMENT")

        if environment_name != "One" and environment_name != "Twenty":
            logging.error("\rINVALID ENVIRONMENT NAME ARGUMENT: {}".format(environment_name))

        relative_file_path = "../env/Reacher_Linux_{environment_name}/Reacher_Linux/Reacher.x86_64"

        if environment_name == "One":
            relative_file_path = relative_file_path.format(environment_name="One")
        if environment_name == "Twenty":
            relative_file_path = relative_file_path.format(environment_name="Twenty")

        current_directory = os.path.dirname(__file__)
        absolut_file_path = os.path.join(current_directory, relative_file_path)

        self._environment = UnityEnvironment(file_name=absolut_file_path, no_graphics=not enable_graphics)

        self._default_brain = self._environment.brains[self._environment.brain_names[0]]

        environment_info = self._environment.reset(train_mode=True)[self._default_brain.brain_name]

        self._agents = environment_info.agents

        self._state = None

    def reset(self, brain: unityagents.brain.BrainParameters = None, train_environment: bool = True) -> NoReturn:
        """
        function to reset environment
        :param brain: brain for which the environment is reset
        :param train_environment: parameter to set whether the environment is for training or for evaluation
        :return: NoReturn
        """
        brain = brain if brain is not None else self._default_brain
        info = self._environment.reset(train_mode=train_environment)[brain.brain_name]
        state = torch.tensor(info.vector_observations, dtype=torch.float)
        self._state = state

    def state(self) -> Tensor:
        """
        function to get the state of the environment
        :return: the state of the environment
        """
        return self._state

    def step(self, action: Tensor, brain: unityagents.brain.BrainParameters = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        function to make a step in the environment
        :param action: action for the agent
        :param brain: brain for which will execute the actions
        :return: state, reward, done following the execution of the action
        """
        brain = brain if brain is not None else self._default_brain

        action = action.tolist()

        info = self._environment.step(action)[brain.brain_name]

        state = torch.tensor(info.vector_observations, dtype=torch.float)
        reward = torch.tensor(info.rewards, dtype=torch.float)
        done = torch.tensor(info.local_done, dtype=torch.float)

        self._state = state

        return state, reward, done

    def close(self) -> NoReturn:
        """
        function to close an environment
        :return: None (to write None on environment on this function call)
        """
        self._environment.close()
        return None

    def number_of_agents(self) -> int:
        """
        function to get the number of agents in the environment
        :return: number of agents in the environment
        """
        return len(self._agents)

    def state_size(self, brain: unityagents.brain.BrainParameters = None) -> int:
        """
        function to get the size of the state vector
        :param brain: brain for which the size of the state vector is returned
        :return: size of the state vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_observation_space_size)

    def action_size(self, brain: unityagents.brain.BrainParameters = None) -> int:
        """
        function to get the size of the action vector
        :param brain: brain for which the size of the action vector is returned
        :return: size of the action vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_action_space_size)

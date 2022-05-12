#!/usr/bin/env python3

import logging
import os

from unityagents import UnityEnvironment

from src.exceptions.invalid_action_excetion import InvalidActionException


class Environment:
    """
    class to wrap the unity environment given for this exercise
    """

    def __init__(self, environment_name, enable_graphics=False):
        """
        constructor for the Environment class
        :param environment_name: name of the environment to use, either "One" (one arm) or "Twenty" (twenty arms)
        :param enable_graphics: parameter to set whether the environment is visualized or not visualized
        """
        if environment_name is None:
            logging.error("MISSING ENVIRONMENT NAME ARGUMENT")

        if environment_name != "One" and environment_name != "Twenty":
            logging.error("INVALID ENVIRONMENT NAME ARGUMENT: {}".format(environment_name))

        relative_file_path = "../env/Reacher_Linux_{environment_name}/Reacher_Linux/Reacher.x86_64"

        if environment_name == "One":
            relative_file_path = relative_file_path.format(environment_name="One")
        if environment_name == "Twenty":
            relative_file_path = relative_file_path.format(environment_name="Twenty")

        current_directory = os.path.dirname(__file__)
        absolut_file_path = os.path.join(current_directory, relative_file_path)

        self._environment = UnityEnvironment(file_name=absolut_file_path, no_graphics=not enable_graphics)

        self._default_brain = self._environment.brains[self._environment.brain_names[0]]

        environment_info = self._environment.reset(train_mode=True)[self._default_brain]
        self._number_of_agents = environment_info.agents

    def reset(self, brain=None, train_environment=True):
        """
        function to reset environment and return environment info
        :param brain: brain for which the environment is reset
        :param train_environment: parameter to set whether the environment is for training or for evaluation
        :return: info about the environment state
        """
        brain = brain if brain is not None else self._default_brain
        info = self._environment.reset(train_mode=train_environment)[brain.brain_name]
        return info

    def get_number_of_agents(self):
        """
        function to get the number of agents in the environment
        :return: number of agents in the environment
        """
        return self._number_of_agents

    def get_state_size(self, brain=None):
        """
        function to get the size of the state vector
        :param brain: brain for which the size of the state vector is returned
        :return: size of the state vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return brain.vector_observation_space_size

    def get_action_size(self, brain=None):
        """
        function to get the size of the action vector
        :param brain: brain for which the size of the action vector is returned
        :return: size of the action vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return brain.vector_action_space_size

    def get_action_range(self):
        """
        function to get the range (as dict containing "min" and "max") for each value of the state vector
        :return: a list of dicts containing "min" and "max"  for each value of the action vector
        """
        return [{"min": -1, "max": 1} for _ in self.get_action_size()]

    def action(self, action_list, brain=None):
        """
        function to set an action for a brain in the environment
        :param action_list: a list of actions for the agents
        :param brain: brain for which the actions are set
        :return: dict containing a dict with lists for the "next_state", the "reward" and the "dones" for each agent
        """
        brain = brain if brain is not None else self._default_brain

        action_list = self._check_action(action_list=action_list)
        info = self._environment.step(action_list)[brain.brain_name]

        return {"next_state": info.vector_observations,
                "reward": info.rewards,
                "dones": info.local_done}

    def _check_action(self, action_list):
        """
        function to check if an action is within the range
        :param action_list: list of actions to check
        :return: list of actions if all action are within the range, raising an exception otherwise
        """
        for action_value, action_range in zip(action_list, self.get_action_range()):
            if action_value < action_range["min"] or action_value > action_range["max"]:
                logging.warning("action value is not in range -1 to 1 (actual value: {})".format(action_value))
            else:
                raise InvalidActionException()

        return action_list

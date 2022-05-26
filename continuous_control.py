#!/usr/bin/env python3

import itertools
import logging
import numpy
import sys
import torch

from collections import OrderedDict

from src.agent import Agent
from src.environment import Environment
from src.hyperparameters.hyperparameters import Hyperparameters
from src.hyperparameters.hyperparameters_range import HyperparametersRange


class ContinuousControl:

    def __init__(self):
        # device variable
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # environment variables
        self._environment = None

        # agent variables
        self._agent = None

        # score variables
        self._absolute_scores = None
        self._average_scores = None

        # hyperparameters for training (see the file ./hyperparameters/hyperparameters.py)
        self._hp: OrderedDict = Hyperparameters().get_dict()

        # hyperparameter range for tuning (see the file ./hyperparameters/hyperparameters_range.py)
        self._hpr: OrderedDict = HyperparametersRange().get_dict()

    def setup(self, mode=None, environment_name=None):
        # setting default mode
        mode = mode if mode is not None else "show"

        # setting default environment
        environment_name = environment_name if environment_name is not None else "Twenty"

        if mode == "train":
            self.train(environment_name=environment_name)
            self.plot()

        if mode == "tune":
            self.tune(environment_name=environment_name)
            self.plot()

        if mode == "show":
            self.show(environment_name=environment_name)

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
        actor_state_size: int = self._environment.get_state_size()
        actor_action_size: int = self._environment.get_action_size()

        critic_state_size: int = self._environment.get_state_size()
        critic_action_size: int = 1

        self._agent = Agent(device=self._device,
                            gamma=self._hp["gamma"],
                            clip=self._hp["clip"])
        self._agent.setup_actor(state_size=actor_state_size,
                                action_size=actor_action_size,
                                layers=self._hp["actor_layers"],
                                activation_function=self._hp["actor_activation_function"],
                                output_function=self._hp["actor_output_function"],
                                optimizer=self._hp["actor_optimizer"],
                                optimizer_learning_rate=self._hp["actor_optimizer_learning_rate"])
        self._agent.setup_critic(state_size=critic_state_size,
                                 action_size=critic_action_size,
                                 layers=self._hp["critic_layers"],
                                 activation_function=self._hp["critic_activation_function"],
                                 output_function=self._hp["critic_output_function"],
                                 optimizer=self._hp["critic_optimizer"],
                                 optimizer_learning_rate=self._hp["critic_optimizer_learning_rate"])

    def train(self, environment_name):
        # setup environment
        enable_graphics = False
        train_environment = True
        self.reset_environment(environment_name=environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)

        self.reset_agent()

        average_scores_per_episode = []
        for episode_index in range(1, self._hp["episodes"] + 1):
            states, actions, logarithmic_probabilities, rewards, future_rewards = self.collect_episode()

            self._agent.update(states, actions, logarithmic_probabilities, future_rewards)

            instance_rewards = []
            for index in range(len(rewards[0])):
                instance_rewards.append([float(run_reward[index]) for run_reward in rewards])

            instance_rewards = [sum(rewards_list) for rewards_list in instance_rewards]
            average_score_for_this_episode = numpy.mean(instance_rewards)
            average_scores_per_episode.append(average_score_for_this_episode)
            logging.info("EPISODE: {} with average reward: {}".format(episode_index, average_score_for_this_episode))

        return average_scores_per_episode

    def tune(self, environment_name):
        for hp_key, hpr_key in zip(self._hp.keys(), self._hpr.keys()):
            if not hp_key == hpr_key:
                logging.error("\rINVALID HYPERPARAMETERS FOR TUNING\n")
                exit()

        hp_iterators = [iter(hpr) for hpr in self._hpr.values()]
        hp_combinations = itertools.product(*hp_iterators)

        best_run_episode_count = None
        best_run_score = None
        best_run_hp = None

        for hp_combination in hp_combinations:
            self._hp = OrderedDict(zip(self._hpr.keys(), hp_combination))
            current_run_scores, current_run_average_scores = self.train(environment_name)

            current_run_episode_count = len(current_run_scores)
            current_run_score = numpy.average(current_run_scores[-100:])
            if best_run_episode_count is None or current_run_episode_count < best_run_episode_count:
                best_run_episode_count = current_run_episode_count
                best_run_score = current_run_score
                best_run_hp = self._hp.copy()

        logging.info("TUNING FINISHED")
        logging.info("EPISODES: {}".format(best_run_episode_count))
        logging.info("SCORE: {}".format(best_run_score))

        ContinuousControl.print_hyperparameters(best_run_hp)

    def show(self, environment_name):
        # setup environment
        enable_graphics = True
        train_environment = False
        self.reset_environment(environment_name=environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)
        # TODO
        pass

    def plot(self):
        # TODO
        pass

    def collect_episode(self):

        states = []
        actions = []
        logarithmic_probabilities = []
        rewards = []

        state = self._environment.reset().vector_observations
        state = torch.tensor(state, dtype=torch.float)
        for step in range(self._hp["steps_per_episode"]):
            action, logarithmic_probability = self._agent.get_action(state)
            environment_info = self._environment.step(action)
            next_state = environment_info["next_state"]
            reward = environment_info["reward"]
            done = environment_info["done"]

            states.append(state)
            actions.append(action)
            logarithmic_probabilities.append(logarithmic_probability)
            rewards.append(reward)

            state = next_state

            if any(done):
                break

        future_rewards = self._agent.get_future_rewards(rewards)

        return states, actions, logarithmic_probabilities, rewards, future_rewards

    @staticmethod
    def print_hyperparameters(hyperparameters):
        for key, value in hyperparameters.items():
            logging.info("{}: {}".format(key, value))


if __name__ == "__main__":
    mode_arg = sys.argv[1]
    environment_name_arg = sys.argv[2]
    ContinuousControl().setup(mode=mode_arg, environment_name=environment_name_arg)

#!/usr/bin/env python3

import itertools
import logging
import numpy
import sys
import torch

from collections import OrderedDict
from matplotlib import pyplot
from typing import NoReturn
from typing import Optional
from typing import List
from typing import Tuple
from torch import Tensor

from src.agent import Agent
from src.environment import Environment
from src.hyperparameters.hyperparameters import Hyperparameters
from src.hyperparameters.hyperparameters_range import HyperparametersRange
from src.utils import Utils


class ContinuousControl:
    """
    class for the continuous control task
    """

    def __init__(self, environment_name) -> None:
        """
        initializer for continuous control class
        :param environment_name: name of the environment to operate
        """
        # device variable
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # environment variables
        self._environment_name = environment_name
        self._environment_graphics = None
        self._environment_training = None

        # environment object
        self._environment: Optional[Environment] = None
        # agent object
        self._agent: Optional[Agent] = None

        # hyperparameters for training (see the file ./hyperparameters/hyperparameters.py)
        self._hp: OrderedDict = Hyperparameters().get_dict()
        # hyperparameter range for tuning (see the file ./hyperparameters/hyperparameters_range.py)
        self._hpr: OrderedDict = HyperparametersRange().get_dict()

    def enable_training(self) -> NoReturn:
        """
        function to enable training mode
            training mode has to be either enabled or disabled
        :return: NoReturn
        """
        self._environment_graphics = False
        self._environment_training = True

    def disable_training(self) -> NoReturn:
        """
        function to disable training mode
            training mode has to be either enabled or disabled
        :return: NoReturn
        """
        self._environment_graphics = True
        self._environment_training = False

    def reset_environment(self) -> NoReturn:
        """
        function to reset the environment
        :return: NoReturn
        """
        if self._environment is None:
            self._environment = Environment(environment_name=self._environment_name,
                                            enable_graphics=self._environment_graphics)

        self._environment.reset(train_environment=self._environment_training)

    def reset_agent(self) -> NoReturn:
        """
        function to reset the agent
        :return: NoReturn
        """
        actor_state_size: int = self._environment.state_size()
        actor_action_size: int = self._environment.action_size()

        critic_state_size: int = self._environment.state_size()
        critic_action_size: int = 1

        self._agent = Agent(device=self._device,
                            clip=self._hp["clip"])
        self._agent.setup_actor(state_size=actor_state_size,
                                action_size=actor_action_size,
                                layers=self._hp["actor_layers"],
                                activation_function=self._hp["actor_activation_function"],
                                output_function=self._hp["actor_output_function"],
                                optimizer=self._hp["actor_optimizer"],
                                optimizer_learning_rate=self._hp["actor_optimizer_learning_rate"],
                                covariance=self._hp["actor_covariance"])
        self._agent.setup_critic(state_size=critic_state_size,
                                 action_size=critic_action_size,
                                 layers=self._hp["critic_layers"],
                                 activation_function=self._hp["critic_activation_function"],
                                 output_function=self._hp["critic_output_function"],
                                 optimizer=self._hp["critic_optimizer"],
                                 optimizer_learning_rate=self._hp["critic_optimizer_learning_rate"])

    def run(self, mode: str) -> NoReturn:
        """
        function to call the function(s) for the passed arguments
        :param mode: operation mode
        :return: NoReturn
        """
        if mode not in ["train", "tune", "show"]:
            logging.error("INVALID OPERATION MODE")

        if mode == "train":
            logging.info("\rSTARTED IN TRAIN MODE")
            scores = self.train(filename="continuous_control.pth")
            self.plot(scores, True, "training.png")

        if mode == "tune":
            logging.info("\rSTARTED IN TUNE MODE")
            scores = self.tune()
            self.plot(scores, True, "tuning.png")

        if mode == "show":
            logging.info("\rSTARTED IN SHOW MODE")
            self.show()

    def train(self, filename: str = None) -> List[float]:
        """
        function to train the agent
        :param filename: filename for the save file of the agent, if not provided not safe file is creat
        :return: scores per episode (score is the mean reward over all agents over  the sum of all steps)
        """
        # enable training mode
        self.enable_training()
        # setup environment
        self.reset_environment()
        self.reset_agent()
        # setup parameters
        episodes = int(self._hp["episodes"])                        # number of episodes
        trajectories = int(self._hp["trajectories"])                # number of trajectories
        steps = int(self._hp["steps"] / self._hp["trajectories"])   # number of steps per trajectories
        training_iterations = int(self._hp["training_iterations"])  # number of training iterations per step

        scores = []         # score of each episode
        for episode in range(1, episodes + 1):
            collected_rewards = []
            self._environment.reset()
            state = self._environment.state().to(device=self._device)
            for trajectory in range(trajectories):
                # collect trajectories and process data
                states, actions, log_probs, rewards, future_rewards = self.collect_trajectory(state=state, steps=steps)
                critics, _ = self._agent.get_critics_and_log_probs(states=states, actions=actions)
                advantages = future_rewards - critics.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                # training
                for _ in range(training_iterations):
                    critics, new_log_probs = self._agent.get_critics_and_log_probs(states=states, actions=actions)
                    self._agent.train_actor(advantages=advantages, new_log_probs=new_log_probs, old_log_probs=log_probs)
                    self._agent.train_critic(critics=critics, future_rewards=future_rewards)

                collected_rewards = collected_rewards + [reward.tolist() for reward in rewards.cpu().numpy()]

            scores.append(self.calculate_score(collected_rewards))
            # logging
            solve = ""
            mean = 0
            if episode < 100:
                solve = "NO MEAN SCORE (OVER 100 EPISODES) YET"
            if episode >= 100:
                mean = numpy.array(scores[-100]).mean()
                solve = "MEAN SCORE (OVER 100 EPISODES): {:3.5f}".format(mean)
            logging.info("\rEPISODE: {:4d} - SCORE: {:3.5f} - {}".format(episode, scores[-1], solve))

            # return if environment is solved
            if mean > 30:
                logging.info("\rENVIRONMENT SOLVED - YEAH!")
                break

        if filename:
            self._agent.save(filename=filename)

        return scores

    def tune(self) -> NoReturn:
        """
        function for hyperparameter tuning
            by executing the train function with different hyperparameter combination the set of hyperparameters solving
            the environment in the least episodes is found and the best scores are returned
        :return: scores of the run with the best set of hyperparameters
        """
        for hp_key, hpr_key in zip(self._hp.keys(), self._hpr.keys()):
            if not hp_key == hpr_key:
                logging.error("\rINVALID HYPERPARAMETERS FOR TUNING")
                exit()

        hp_iterators = [iter(hpr) for hpr in self._hpr.values()]
        hp_combinations = itertools.product(*hp_iterators)

        best_run_scores = None
        best_run_episodes = None
        best_run_hp = None

        for hp_combination in hp_combinations:
            self._hp = OrderedDict(zip(self._hpr.keys(), hp_combination))

            logging.info("----------------------------------------------------------------------")
            self.print_hyperparameters(self._hp)
            logging.info("----------------------------------------------------------------------")

            current_run_scores = self.train()
            current_run_episodes = len(current_run_scores)

            if best_run_episodes is None or current_run_episodes < best_run_episodes:
                best_run_scores = current_run_scores
                best_run_episodes = current_run_episodes
                best_run_hp = self._hp.copy()

        best_run_episodes = len(best_run_episodes)
        best_run_mean_score = numpy.average(best_run_episodes[-100:])

        logging.info("TUNING FINISHED")
        logging.info("BEST RUN EPISODES: {}".format(best_run_episodes))
        logging.info("BEST RUN MEAN SCORE (OVER 100 EPISODES): {}".format(best_run_mean_score))

        ContinuousControl.print_hyperparameters(best_run_hp)

        self._environment.close()

        return best_run_scores

    def show(self) -> NoReturn:
        """
        function to showcase the saved agent
        :return: NoReturn
        """
        # disable training mode
        self.disable_training()
        # setup environment
        self.reset_environment()
        self.reset_agent()

        self._agent = self._agent.load("continuous_control.pth")
        state = self._environment.state()
        for step in range(10000):
            state = state.to(device=self._device)
            action, _ = self._agent.get_action_and_log_prob(state=state)
            action = action.to(device=self._device)
            state, _, done = self._environment.step(action=action)
            if any(done):
                break

        self._environment.close()

    def collect_trajectory(self, state: Tensor, steps: int, gamma: float = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        function to collect a trajectory
        :param state: state of the environment for the trajectory start
        :param steps: number of steps in the trajectory
        :param gamma: discount factor for the future rewards
        :return: a tuple of states, actions, log_probs, rewards, future_rewards
        """
        gamma = gamma if gamma is not None else self._hp["gamma"]

        states = []
        actions = []
        log_probs = []
        rewards = []
        future_rewards = None

        for step in range(steps):
            action, log_prob = self._agent.get_action_and_log_prob(state=state)
            follow_up_state, reward, done = self._environment.step(action=action)

            states.append(state.cpu().numpy())
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            rewards.append(reward.cpu().numpy())

            state = follow_up_state.to(device=self._device)

        future_rewards = Utils.calculate_future_rewards(rewards=rewards, gamma=gamma)

        states = torch.tensor(numpy.array(states), dtype=torch.float).to(device=self._device)
        actions = torch.tensor(numpy.array(actions), dtype=torch.float).to(device=self._device)
        log_probs = torch.tensor(numpy.array(log_probs), dtype=torch.float).to(device=self._device)
        rewards = torch.tensor(numpy.array(rewards), dtype=torch.float).to(device=self._device)
        future_rewards = torch.tensor(numpy.array(future_rewards), dtype=torch.float).to(device=self._device)
        return states, actions, log_probs, rewards, future_rewards

    @staticmethod
    def calculate_score(rewards: List[List[float]]) -> float:
        """
        function to calculate the score (score is the mean reward over all agents over the sum of all steps)
        :param rewards: list of rewards per agent per step
        :return: score (score is the mean reward over all agents over the sum of all steps)
        """
        total_agent_reward = [0 for _ in range(len(rewards[0]))]

        for reward in rewards:
            for index, agent_reward in enumerate(reward):
                total_agent_reward[index] += float(agent_reward)

        return numpy.array(total_agent_reward).mean()

    @staticmethod
    def plot(scores: List[float], show: bool = False, filename: str = None) -> NoReturn:
        """
        function to create a plot of the given scores
        :param scores: scores to plot
        :param show: if true the plot will be shown
        :param filename: if not None the plot will be stored to the given destination
        :return: NoReturn
        """
        fig = pyplot.figure()
        pyplot.plot(numpy.arange(len(scores)), scores)
        pyplot.ylabel('Score')
        pyplot.xlabel('Episode')
        if show:
            pyplot.show()
        if filename is not None:
            pyplot.savefig(filename)

    @staticmethod
    def print_hyperparameters(hyperparameters: OrderedDict) -> NoReturn:
        """
        function to print the given hyperparameters
        :param hyperparameters: hyperparameters to print
        :return: NoReturn
        """
        for key, value in hyperparameters.items():
            logging.info("{}: {}".format(key, value))


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    # ARG 2: ENVIRONMENT NAME
    # ARG 3: SAVE AGENT FILENAME
    mode_arg = sys.argv[1]
    environment_name_arg = sys.argv[2]
    ContinuousControl(environment_name=environment_name_arg).run(mode=mode_arg)

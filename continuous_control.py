#!/usr/bin/env python3

import itertools
import logging
import numpy
import sys
import torch

from collections import OrderedDict, deque

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
            # self.plot()

        if mode == "tune":
            pass
            # self.tune(environment_name=environment_name)
            # self.plot()

        if mode == "show":
            pass
            # self.show(environment_name=environment_name)

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
                            decay=self._hp["decay"],
                            clip=self._hp["clip"],
                            kl_difference_limit=self._hp["kl_difference_limit"])
        self._agent.setup_actor(state_size=actor_state_size,
                                action_size=actor_action_size,
                                layers=self._hp["actor_layers"],
                                activation_function=self._hp["actor_activation_function"],
                                output_function=self._hp["actor_output_function"],
                                training_iterations=self._hp["actor_training_iterations"],
                                optimizer=self._hp["actor_optimizer"],
                                optimizer_learning_rate=self._hp["actor_optimizer_learning_rate"],
                                covariance=self._hp["actor_covariance"])
        self._agent.setup_critic(state_size=critic_state_size,
                                 action_size=critic_action_size,
                                 layers=self._hp["critic_layers"],
                                 activation_function=self._hp["critic_activation_function"],
                                 output_function=self._hp["critic_output_function"],
                                 training_iterations=self._hp["critic_training_iterations"],
                                 optimizer=self._hp["critic_optimizer"],
                                 optimizer_learning_rate=self._hp["critic_optimizer_learning_rate"])

    def train(self, environment_name, episodes: int = None, rollouts: int = None, steps: int = None):
        episodes = episodes if episodes is not None else self._hp["episodes"]
        rollouts = rollouts if rollouts is not None else self._hp["rollouts"]
        steps = steps if steps is not None else self._hp["steps"]

        trajectories_per_episode = self._hp["trajectories_per_episode"]
        steps = self._hp["steps_per_trajectory"]

        # setup environment
        enable_graphics = False
        train_environment = True
        self.reset_environment(environment_name=environment_name,
                               enable_graphics=enable_graphics,
                               train_environment=train_environment)

        self.reset_agent()

        # TODO: track rewards of single agents over each episode
        episode_scores = []
        for episode in range(episodes):
            episode_rewards = []
            state = self._environment.reset().to(device=self._device)
            for trajectory in range(trajectories_per_episode):
                states, actions, log_probs, rewards, rewards_to_go = self.collect_trajectory(state=state,
                                                                                             steps=steps)
                critics, _ = self._agent.get_critics_and_log_probs(states=states,
                                                                   actions=actions)
                advantages = rewards_to_go - critics.detach()
                normalized_advantages = ( advantages - advantages.mean() ) / ( advantages.std() + 1e-10 )

                self._agent.train_agent(states=states,
                                        actions=actions,
                                        rewards_to_go=rewards_to_go,
                                        old_log_probs=log_probs,
                                        advantages=normalized_advantages)

                rewards = rewards.cpu().numpy()
                for reward in rewards:
                    episode_rewards.append(reward.tolist())

            episode_score = self.calculate_score(episode_rewards)
            episode_scores.append(episode_score)

            if len(episode_scores) < 100:
                print("episode: {:4d} - score: {:3.5f} - below 100 iterations".format(episode, self.calculate_score(episode_rewards)))

            if len(episode_scores) >= 100:
                temp = reversed(episode_scores)
                mean_over_100 = numpy.array(temp[0:99]).mean()
                print("episode: {:4d} score: {:3.5f} - mean over 100: {}".format(episode, self.calculate_score(
                    episode_rewards, mean_over_100)))

    def calculate_score(self, rewards):
        total_agent_reward = [0 for _ in range(self._environment.number_of_agents())]

        for reward in rewards:
            for index, agent_reward in enumerate(reward):
                total_agent_reward[index] += float(agent_reward)

        return numpy.array(total_agent_reward).mean()

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

    def rollout(self, state: torch.Tensor, steps: int):
        states = []
        actions = []
        rewards = []
        log_probs = []

        for _ in range(steps):
            action, actual_log_prob = self._agent.get_action_and_log_prob(state)

            next_state, reward, done = self._environment.step(action)
            next_state = next_state.to(self._device)
            reward = reward.to(self._device)
            done = done.to(self._device)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(actual_log_prob)

            state = next_state

            if any(done):
                break

        return states, actions, rewards, log_probs

    def collect_trajectory(self, state: torch.Tensor, steps: int):
        states = []
        actions = []
        log_probs = []
        rewards = []

        for step in range(steps):
            action, log_prob = self._agent.get_action_and_log_prob(state=state)
            follow_up_state, reward, done = self._environment.step(action=action)

            states.append(state.cpu().numpy())
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            rewards.append(reward.cpu().numpy())

            state = follow_up_state.to(device=self._device)
        # TODO: calculate rewards to go (and rename them)
        rewards_to_go = self._agent.calculate_rewards_to_go(rewards)

        states = torch.tensor(numpy.array(states), dtype=torch.float).to(device=self._device)
        actions = torch.tensor(numpy.array(actions), dtype=torch.float).to(device=self._device)
        log_probs = torch.tensor(numpy.array(log_probs), dtype=torch.float).to(device=self._device)
        rewards = torch.tensor(numpy.array(rewards), dtype=torch.float).to(device=self._device)
        rewards_to_go = torch.tensor(numpy.array(rewards_to_go), dtype=torch.float).to(device=self._device)
        return states, actions, log_probs, rewards, rewards_to_go

    @staticmethod
    def print_hyperparameters(hyperparameters):
        for key, value in hyperparameters.items():
            logging.info("{}: {}".format(key, value))


if __name__ == "__main__":
    mode_arg = sys.argv[1]
    environment_name_arg = sys.argv[2]
    ContinuousControl().setup(mode=mode_arg, environment_name=environment_name_arg)

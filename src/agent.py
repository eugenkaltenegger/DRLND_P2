#!/usr/bin/env python3

import logging
import torch

from typing import Tuple
from typing import NoReturn

from torch import Tensor

from src.model import Model


class Agent:
    """
    class representing an PPO agent (with actor and critic)
    """

    def __init__(self, device: torch.device, gamma: float, clip: float):
        """
        constructor for agent class
            this constructor requires the setup_actor and setup_critic function afterwards
        :param device:
        :param gamma:
        :param clip:
        """
        super(Agent, self).__init__()

        self._device = device

        self._gamma = gamma
        self._clip = clip

        self._actor = None
        self._actor_optimizer = None
        self._actor_training_iterations = None
        self._covariance = None
        self._covariance_matrix = None

        self._critic = None
        self._critic_optimizer = None
        self._critic_training_iterations = None

    def setup_actor(self,
                    state_size: int,
                    action_size: int,
                    layers: [int],
                    activation_function: torch.nn.Module,
                    output_function: torch.nn.Module,
                    optimizer: torch.optim,
                    optimizer_learning_rate: float,
                    covariance: int) -> NoReturn:
        """
        function to set up the actor
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :param optimizer: optimizer for the model
        :param optimizer_learning_rate: learning rate of the optimizer
        :param covariance: covariance of sample distribution
        :return:
        """
        self._actor = Model().setup(state_size=state_size,
                                    action_size=action_size,
                                    layers=layers,
                                    activation_function=activation_function,
                                    output_function=output_function)
        self._actor = self._actor.to(device=self._device)
        self._actor_optimizer = optimizer(params=self._actor.parameters(), lr=optimizer_learning_rate)

        self._covariance = torch.full(size=(action_size,), fill_value=covariance).to(device=self._device)
        self._covariance_matrix = torch.diag(self._covariance).to(device=self._device)

    def setup_critic(self,
                     state_size: int,
                     action_size: int,
                     layers: [int],
                     activation_function: torch.nn.Module,
                     output_function: torch.nn.Module,
                     optimizer: torch.optim,
                     optimizer_learning_rate: float) -> NoReturn:
        """
        function to set up the critic
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :param optimizer: optimizer for the model
        :param optimizer_learning_rate: learning rate of the optimizer
        :return:
        """
        self._critic = Model().setup(state_size=state_size,
                                     action_size=action_size,
                                     layers=layers,
                                     activation_function=activation_function,
                                     output_function=output_function)
        self._critic = self._critic.to(device=self._device)
        self._critic_optimizer = optimizer(params=self._critic.parameters(), lr=optimizer_learning_rate)

    def get_action_and_log_prob(self,
                                state: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self._actor(state)
        distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=self._covariance_matrix)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.detach(), log_probability.detach()

    def get_critics_and_log_probs(self,
                                  states: Tensor,
                                  actions: Tensor) -> Tuple[Tensor, Tensor]:
        critics = self._critic(states)
        means = self._actor(states)
        distributions = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=self._covariance_matrix)
        log_probs = distributions.log_prob(actions)
        critics = critics.squeeze()  # Tensor:(..., 20, 1) -> Tensor:(..., 20)
        return critics, log_probs

    def train_agent(self,
                    states: Tensor,
                    actions: Tensor,
                    rewards_to_go: Tensor,
                    old_log_probs: Tensor,
                    advantages: Tensor,
                    training_iterations: int) -> NoReturn:
        for _ in range(training_iterations):
            critics, new_log_probs = self.get_critics_and_log_probs(states=states, actions=actions)
            self.train_actor(advantages=advantages,
                             new_log_probs=new_log_probs,
                             old_log_probs=old_log_probs)
            self.train_critic(critics=critics,
                              future_rewards=rewards_to_go)

    def train_actor(self, advantages: Tensor, new_log_probs: Tensor, old_log_probs: Tensor) -> NoReturn:
        """

        :param advantages:
        :param new_log_probs:
        :param old_log_probs:
        :return:
        """
        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_policy_ratio = policy_ratio.clamp(1 - self._clip, 1 + self._clip)

        full_loss = policy_ratio * advantages
        clipped_loss = clipped_policy_ratio * advantages

        loss = -torch.min(full_loss, clipped_loss).mean()
        Agent.optimize(optimizer=self._actor_optimizer, loss=loss)

    def train_critic(self, critics: Tensor, future_rewards: Tensor) -> NoReturn:
        """

        :param critics:
        :param future_rewards:
        :return:
        """
        loss = (future_rewards - critics).pow(2).mean()
        Agent.optimize(optimizer=self._critic_optimizer, loss=loss)

    @staticmethod
    def optimize(optimizer: torch.optim, loss: Tensor, retain_graph: bool = True) -> NoReturn:
        """
        function to optimize a model
        :param optimizer: optimizer for a model
        :param loss: loss for optimization
        :param retain_graph: boolean flag for retaining the graph
        :return: NoReturn
        """
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

    def save(self, filename: str = None) -> NoReturn:
        """
        function to save a checkpoint of the agent
        :param filename: name of the checkpoint file
        :return: NoReturn
        """
        filename = filename if filename is not None else "checkpoint.pth"
        checkpoint = {"actor": self._actor.to_checkpoint(),
                      "critic": self._critic.to_checkpoint()}
        torch.save(checkpoint, filename)
        logging.info("agent saved to checkpoint (file: {})".format(filename))

    def load(self, filename: str = None) -> NoReturn:
        """
        function to load a checkpoint of the agent
        :param filename: name of the checkpoint file
        :return: self
        """
        filename = filename if filename is not None else "checkpoint.pth"
        checkpoint = torch.load(filename)
        self._actor = Model().from_checkpoint(checkpoint=checkpoint["actor"])
        self._critic = Model().from_checkpoint(checkpoint=checkpoint["critic"])
        logging.info("agent loaded from checkpoint (file: {})".format(filename))

        return self

#!/usr/bin/env python3

import logging
import torch
import typing

from torch import Tensor
from typing import NoReturn
from typing import Tuple

from src.network import Network

# required for typehint Self
Self = typing.TypeVar("Self", bound="Agent")


class Agent:
    """
    class to represent a PPO agent (with actor and critic)
    """

    def __init__(self, device: torch.device, clip: float = None) -> None:
        """
        initializer for agent class
            this constructor requires the setup_actor and setup_critic function afterwards
        :param device:
        :param clip:
        """
        super(Agent, self).__init__()

        self._device = device

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
        :return: NoReturn
        """
        self._actor = Network().setup(state_size=state_size,
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
        :return: NoReturn
        """
        self._critic = Network().setup(state_size=state_size,
                                       action_size=action_size,
                                       layers=layers,
                                       activation_function=activation_function,
                                       output_function=output_function)
        self._critic = self._critic.to(device=self._device)
        self._critic_optimizer = optimizer(params=self._critic.parameters(), lr=optimizer_learning_rate)

    def get_action_and_log_prob(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        function to utilize the actor network
        :param state: the state of the environment
        :return: a sample action and the according logarithmic probability
        """
        mean = self._actor(state)
        distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=self._covariance_matrix)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.detach(), log_probability.detach()

    def get_critics_and_log_probs(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        function to utilize the critic network
        :param states: the states of the environment
        :param actions: the actions of the agent (according to the states)
        :return: critics on the actions and the logarithmic probability of the actions
        """
        critics = self._critic(states)
        means = self._actor(states)
        distributions = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=self._covariance_matrix)
        log_probs = distributions.log_prob(actions)
        critics = critics.squeeze()  # Tensor:(..., 20, 1) -> Tensor:(..., 20)
        return critics, log_probs

    def train_actor(self, advantages: Tensor, new_log_probs: Tensor, old_log_probs: Tensor) -> NoReturn:
        """
        function to train the actor network
        :param advantages: advantages
        :param new_log_probs: new logarithmic probabilities
        :param old_log_probs: old logarithmic probabilities
        :return: NoReturn
        """
        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_policy_ratio = policy_ratio.clamp(1 - self._clip, 1 + self._clip)

        full_loss = policy_ratio * advantages
        clipped_loss = clipped_policy_ratio * advantages

        loss = -torch.min(full_loss, clipped_loss).mean()
        Agent.optimize(optimizer=self._actor_optimizer, loss=loss)

    def train_critic(self, critics: Tensor, future_rewards: Tensor) -> NoReturn:
        """
        function to train the critic network
        :param critics: critics
        :param future_rewards: future rewards (already discounted)
        :return: NoReturn
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
        checkpoint = {"actor": self._actor.to_checkpoint_dict(),
                      "critic": self._critic.to_checkpoint_dict()}
        torch.save(checkpoint, filename)
        logging.info("\rAGENT SAVED (file: {})".format(filename))

    def load(self, filename: str = None) -> Self:
        """
        function to load a checkpoint of the agent
        :param filename: name of the checkpoint file
        :return: self
        """
        filename = filename if filename is not None else "checkpoint.pth"
        checkpoint = torch.load(filename)
        self._actor = Network().from_checkpoint(checkpoint=checkpoint["actor"])
        self._critic = Network().from_checkpoint(checkpoint=checkpoint["critic"])
        logging.info("agent loaded from checkpoint (file: {})".format(filename))

        return self

#!/usr/bin/env python3

import logging
from typing import Tuple, List
from typing import NoReturn

import numpy
import torch

from torch.distributions import Categorical

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
                                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._actor(state)
        distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=self._covariance_matrix)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.detach(), log_probability.detach()

    def get_critics_and_log_probs(self,
                                  states: torch.Tensor,
                                  actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        critics = self._critic(states)
        means = self._actor(states)
        distributions = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=self._covariance_matrix)
        log_probs = distributions.log_prob(actions)
        critics = critics.squeeze()  # Tensor:(..., 20, 1) -> Tensor:(..., 20)
        return critics, log_probs

    def train_agent(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards_to_go: torch.Tensor,
                    old_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    training_iterations: int) -> NoReturn:
        for _ in range(training_iterations):
            critics, new_log_probs = self.get_critics_and_log_probs(states=states, actions=actions)
            self.train_actor(advantages=advantages,
                             new_log_probs=new_log_probs,
                             old_log_probs=old_log_probs)
            self.train_critic(critics=critics,
                              future_rewards=rewards_to_go)

    def train_actor(self,
                    advantages: torch.Tensor,
                    new_log_probs: torch.Tensor,
                    old_log_probs: torch.Tensor) -> NoReturn:

        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_policy_ratio = policy_ratio.clamp(1 - self._clip, 1 + self._clip)

        full_loss = policy_ratio * advantages
        clipped_loss = clipped_policy_ratio * advantages

        policy_loss = -torch.min(full_loss, clipped_loss).mean()

        self._actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

    def train_critic(self,
                     critics: torch.Tensor,
                     future_rewards: torch.Tensor) -> NoReturn:
        loss = (future_rewards - critics).pow(2).mean()

        self._critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self._critic_optimizer.step()

    def calculate_discounts(self,
                            rewards: List[torch.Tensor],
                            gamma: float = None) -> [torch.Tensor]:
        gamma = gamma if gamma is not None else self._gamma

        rewards.reverse()
        discount = torch.tensor([0.0 for _ in range(len(rewards[0]))]).to(device=self._device)
        discounts = []
        for reward in rewards:
            discount = reward + discount * gamma
            discounts.append(discount)

        discounts.reverse()

        return discounts

    def calculate_advantages(self,
                             rewards: torch.Tensor,
                             critics: torch.Tensor,
                             gamma: float = None,
                             decay: float = None) -> torch.Tensor:
        gamma = gamma if gamma is not None else self._gamma

        rewards = rewards.cpu().numpy()
        rewards = [item for item in rewards]
        critics = critics.cpu().numpy()
        critics = [item for sublist in critics for item in sublist]

        next_values = numpy.concatenate([critics[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, critics, next_values)]

        advantages = [deltas[-1]]
        for i in reversed(range(len(deltas) - 1)):
            advantages.append(deltas[i] + decay * gamma * advantages[-1])

        advantages.reverse()
        advantages = torch.tensor(advantages, dtype=torch.float).to(device=self._device)
        return advantages

    def calculate_future_rewards(self, rewards, gamma: float = None):
        gamma = gamma if gamma is not None else self._gamma
        future_rewards = []

        discounted_reward = 0

        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * gamma
            future_rewards.insert(0, discounted_reward)

        return future_rewards

    def save(self, filename: str = None) -> NoReturn:
        """
        function to save a checkpoint of the agent
        :param filename: name of the checkpoint file
        :return:
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

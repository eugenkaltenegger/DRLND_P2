#!/usr/bin/env python3

import logging
from typing import Tuple
from typing import NoReturn

import numpy
import torch

from torch.distributions import Categorical

from src.model import Model


class Agent(torch.nn.Module):
    """
    class representing an PPO agent (with actor and critic)
    """

    def __init__(self, device: torch.device, gamma: float, decay: float, clip: float, kl_difference_limit: float):
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
        self._decay = decay
        self._clip = clip
        self._kl_difference_limit = kl_difference_limit

        self._actor = None
        self._actor_optimizer = None
        self._actor_training_iterations = None

        self._critic = None
        self._critic_optimizer = None
        self._critic_training_iterations = None

    def setup_actor(self,
                    state_size: int,
                    action_size: int,
                    layers: [int],
                    activation_function: torch.nn.Module,
                    output_function: torch.nn.Module,
                    training_iterations: int,
                    optimizer: torch.optim,
                    optimizer_learning_rate: float) -> NoReturn:
        """
        function to set up the actor
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :param training_iterations: number of iterations of training with a given sample
        :param optimizer: optimizer for the model
        :param optimizer_learning_rate: learning rate of the optimizer
        :return:
        """
        self._actor = Model().setup(state_size=state_size,
                                    action_size=action_size,
                                    layers=layers,
                                    activation_function=activation_function,
                                    output_function=output_function)
        self._actor = self._actor.to(device=self._device)
        self._actor_training_iterations = training_iterations
        self._actor_optimizer = optimizer(params=self._actor.parameters(), lr=optimizer_learning_rate)

    def setup_critic(self,
                     state_size: int,
                     action_size: int,
                     layers: [int],
                     activation_function: torch.nn.Module,
                     output_function: torch.nn.Module,
                     training_iterations: int,
                     optimizer: torch.optim,
                     optimizer_learning_rate: float) -> NoReturn:
        """
        function to set up the critic
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
        :param training_iterations: number of iterations of training with a given sample
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
        self._critic_training_iterations = training_iterations
        self._critic_optimizer = optimizer(params=self._critic.parameters(), lr=optimizer_learning_rate)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.get_action(states=states)
        critics = self.get_critic(states=states)
        return actions, critics

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self._device)
        return self._actor(states)

    def get_critic(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self._device)
        return self._critic(states)

    def train_agent(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    advantages: torch.Tensor,
                    old_log_probabilities: torch.Tensor) -> NoReturn:
        # TODO: check of discount is handed correctly (track whole route of discount and rewards)
        discount = self.calculate_discount(rewards=rewards)
        self.train_actor(states=states,
                         actions=actions,
                         advantages=advantages,
                         old_log_probabilities=old_log_probabilities)
        self.train_critic(states=states,
                          discount=discount)

    def train_actor(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    advantages: torch.Tensor,
                    old_log_probabilities: torch.Tensor) -> NoReturn:
        for _ in range(self._actor_training_iterations):
            # TODO: probably same problem with logits as in rollout
            logits = self._actor(states)
            logits = Categorical(logits=logits)
            new_log_probabilities = logits.log_prob(actions)

            policy_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            clipped_policy_ratio = policy_ratio.clamp(1 - self._clip, 1 + self._clip)

            full_loss = policy_ratio * advantages
            clipped_loss = clipped_policy_ratio * advantages

            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            self._actor_optimizer.zero_grad()
            # TODO check if retain_graph or retain_graph
            policy_loss.backward(keep_graph=True)
            # policy_loss.backward(retain_graph=True)
            self._actor_optimizer.step()

            kl_difference = (old_log_probabilities - new_log_probabilities).mean()
            if kl_difference > self._kl_difference_limit:
                break

    def train_critic(self,
                     states: torch.Tensor,
                     discount: torch.Tensor) -> NoReturn:
        for _ in range(self._critic_training_iterations):
            critics = self._critic(states)
            loss = ((discount - critics) ** 2).mean()

            self._critic_optimizer.zero_grad()
            # TODO check if retain_graph or retain_graph
            loss.backward(keep_graph=True)
            # loss.backward(retain_graph=True)
            self._critic_optimizer.step()

    def calculate_discount(self,
                           rewards: torch.Tensor,
                           gamma: float = None) -> torch.Tensor:
        gamma = gamma if gamma is not None else self._gamma

        rewards_list = rewards.tolist()
        discounted_rewards = [rewards_list[-1]]
        for i in reversed(range(len(rewards_list) - 1)):
            discounted_rewards.append(float(rewards[i]) + gamma * discounted_rewards[-1])
        discounted_rewards.reverse()
        return torch.tensor(discounted_rewards, dtype=torch.float)

    def calculate_advantages(self,
                             rewards: torch.Tensor,
                             critics: torch.Tensor,
                             gamma: float = None,
                             decay: float = None) -> torch.Tensor:
        gamma = gamma if gamma is not None else self._gamma
        decay = decay if decay is not None else self._decay

        rewards = rewards.cpu().detach().numpy()
        rewards = [item for item in rewards]
        critics = critics.cpu().detach().numpy()
        critics = [item for sublist in critics for item in sublist]

        next_values = numpy.concatenate([critics[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, critics, next_values)]

        advantages = [deltas[-1]]
        for i in reversed(range(len(deltas) - 1)):
            advantages.append(deltas[i] + decay * gamma * advantages[-1])

        advantages.reverse()
        return torch.tensor(advantages, dtype=torch.float).to(device=self._device)

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

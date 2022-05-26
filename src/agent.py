#!/usr/bin/env python3

import logging
from typing import Tuple, List
from typing import NoReturn

import torch

from torch.distributions import MultivariateNormal, Normal

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
        self._device = device

        self._gamma = gamma
        self._clip = clip

        self._actor = None
        self._actor_optimizer = None

        self._covariance_variance = None
        self._covariance_matrix = None

        self._critic = None
        self._critic_optimizer = None

    def setup_actor(self,
                    state_size: int,
                    action_size: int,
                    layers: [int],
                    activation_function: torch.nn.Module,
                    output_function: torch.nn.Module,
                    optimizer: torch.optim,
                    optimizer_learning_rate: float) -> NoReturn:
        """
        function to set up the actor
        :param state_size: size of the state vector / input vector
        :param action_size: size of the action vector / output vector
        :param layers: list of layer sizes
        :param activation_function: function to process output after each layer (except last)
        :param output_function: function to process output after last layer
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
        self._actor_optimizer = optimizer(params=self._actor.parameters(), lr=optimizer_learning_rate)

        self._covariance_variance = torch.full(size=(action_size,), fill_value=0.25).to(device=self._device)
        self._covariance_matrix = torch.diag(self._covariance_variance).to(device=self._device)

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

    def get_future_rewards(self, rewards: List[torch.Tensor], gamma=None) -> List[torch.Tensor]:
        # TODO: documentation
        gamma = gamma if gamma is not None else self._gamma

        rewards_array = [rewards.numpy() for rewards in rewards]

        rewards_per_instance = []
        for instance in range(len(rewards_array[0])):
            rewards_per_instance.append([reward_array[instance] for reward_array in rewards_array])

        future_rewards_per_instance = []
        for rewards in rewards_per_instance:
            future_rewards_per_instance.append(self._get_future_rewards_for_single_instance(rewards=rewards, gamma=gamma))

        future_rewards_per_run = []
        for run in range(len(future_rewards_per_instance[0])):
            future_rewards_per_run.append([rewards_list[run] for rewards_list in future_rewards_per_instance])

        rewards_tensor_list = [torch.tensor(run, dtype=torch.float) for run in future_rewards_per_run]
        return rewards_tensor_list

    @staticmethod
    def _get_future_rewards_for_single_instance(rewards: List, gamma: float) -> List:
        rewards.reverse()
        future_rewards = []
        discounted_rewards = 0
        for reward in rewards:
            discounted_rewards = reward + gamma * discounted_rewards
            future_rewards.append(discounted_rewards)
        future_rewards.reverse()
        return future_rewards

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: documentation
        state = state.to(self._device)

        # self._covariance_fill = max(self._covariance_fill * 0.999, 0.1)
        # self._covariance_variance = torch.full(size=(self._action_size,), fill_value=self._covariance_fill).to(device=self._device)
        # self._covariance_matrix = torch.diag(self._covariance_variance).to(device=self._device)

        # mean = self._actor(state)
        # deviation = torch.nn.Softplus(torch.nn.Parameter(torch.zeros()))
        #
        # distribution = Normal(self._actor(state), deviation)
        distribution = MultivariateNormal(self._actor(state), self._covariance_matrix)
        action = distribution.sample()
        logarithmic_probability = distribution.log_prob(action)
        return action.detach(), logarithmic_probability.detach()

    def get_critic(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: documentation
        # states = [state.to(self._device) for state in states]
        # actions = [action.to(self._device) for action in actions]
        states = states.to(self._device)
        actions = actions.to(self._device)

        critic = self._critic(states).squeeze()
        distribution = MultivariateNormal(self._actor(states), self._covariance_matrix)
        logarithmic_probability = distribution.log_prob(actions)
        return critic, logarithmic_probability

    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               logarithmic_probabilities: torch.Tensor,
               future_rewards: torch.Tensor) -> NoReturn:

        states = [state.to(self._device) for state in states]
        actions = [action.to(self._device) for action in actions]
        logarithmic_probabilities = [lp.to(self._device) for lp in logarithmic_probabilities]
        future_rewards = [future_reward.to(self._device) for future_reward in future_rewards]

        runs = zip(states, actions, logarithmic_probabilities, future_rewards)

        for run in runs:
            state = run[0]
            action = run[1]
            logarithmic_probability = run[2]
            future_reward = run[3]

            critic, curr_log_probs = self.get_critic(state, action)
            advantage = future_reward - critic.detach()  # ALG STEP 5
            # normalizing advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            ratios = torch.exp(curr_log_probs - logarithmic_probability)

            # calculate surrogate
            surrogate_1 = ratios * advantage
            surrogate_2 = torch.clamp(ratios, 1 - self._clip, 1 + self._clip) * advantage

            # calculate actor loss and critic loss
            actor_loss = (-torch.min(surrogate_1, surrogate_2)).mean()
            critic_loss = torch.nn.MSELoss()(critic, future_reward)

            # Calculate gradients and perform backward propagation for actor network
            self._actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self._actor_optimizer.step()

            # Calculate gradients and perform backward propagation for critic network
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()

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

#!/usr/bin/env python3

from typing import List


class Utils:
    """
    class to hold machine learning utilities
    """
    
    @staticmethod
    def calculate_future_rewards(rewards: List, gamma: float):
        """
        function to calculate the future rewards with discounts
        :param rewards: rewards
        :param gamma: discount factor
        :return: future rewards with discount
        """
        future_rewards = []

        discounted_reward = 0

        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * gamma
            future_rewards.insert(0, discounted_reward)

        return future_rewards

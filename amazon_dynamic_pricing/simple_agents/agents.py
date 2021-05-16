import numpy as np


class RandomAgent:
    def __init__(self, min_price, max_price):
        self.min_price = min_price
        self.max_price = max_price

    def act(self, observation):
        return self.min_price + (self.max_price - self.min_price) * np.random.rand()


class ConstantAgent:
    def __init__(self, price):
        self.price = price

    def act(self, observation):
        return self.price


class GreedyAgent:
    def __init__(self, min_price, max_price, demand_model, price_rank_coefficients):
        self.min_price = min_price
        self.max_price = max_price
        self.demand_model = demand_model
        self.price_to_rank_fn = np.poly1d(price_rank_coefficients)

    def act(self, observation):
        greedy_price = None
        highest_revenue = -np.Inf
        prices = np.linspace(self.min_price, self.max_price, 10)

        for price in prices:
            rank = self.price_to_rank_fn(price)
            demand_model_input = np.concatenate((observation[:2], np.array([rank]), observation[2:-2]))
            expected_sales = self.demand_model.predict(demand_model_input[None, :])[0]
            expected_revenue = expected_sales * price

            if expected_revenue > highest_revenue:
                highest_revenue = expected_revenue
                greedy_price = price

        return greedy_price

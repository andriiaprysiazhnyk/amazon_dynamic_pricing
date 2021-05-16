import os
import pickle
import numpy as np


# base agent, that implements basic operations with states and actions
class TabularAgent:
    def __init__(self, min_price, max_price, bins_number):
        self.min_price = min_price
        self.max_price = max_price
        self.min_sales = 0
        self.max_sales = 606
        self.min_rank = 7156
        self.max_rank = 330080
        self.min_adspend = 0
        self.max_adspend = 555
        self.min_inventory = 0
        self.max_inventory = 29470
        self.min_day = 0
        self.max_day = 30
        self.bins_number = bins_number
        self.price_division = (self.max_price - self.min_price) / self.bins_number
        self.sales_division = (self.max_sales - self.min_sales) / self.bins_number
        self.rank_division = (self.max_rank - self.min_rank) / self.bins_number
        self.adspend_division = (self.max_adspend - self.min_adspend) / self.bins_number
        self.inventory_division = (self.max_inventory - self.min_inventory) / self.bins_number
        self.day_division = (self.max_day - self.min_day) / self.bins_number
        self.sizes = [bins_number, bins_number, bins_number, bins_number, bins_number, bins_number,
                      bins_number, bins_number, bins_number, bins_number, bins_number, bins_number,
                      bins_number, 2]

    def _correct(self, bin_number):
        bin_number = min(bin_number, self.bins_number - 1)
        bin_number = max(0, bin_number)
        return bin_number

    def _discretize_price(self, price):
        return self._correct(int((price - self.price_division - self.min_price) // self.price_division))

    def _discretize_sales(self, sales):
        return self._correct(int((sales - self.sales_division - self.min_sales) // self.sales_division))

    def _discretize_rank(self, rank):
        return self._correct(int((rank - self.rank_division - self.min_rank) // self.rank_division))

    def _discretize_adspend(self, adspend):
        return self._correct(int((adspend - self.adspend_division - self.min_adspend) // self.adspend_division))

    def _discretize_inventory(self, inventory):
        return self._correct(int((inventory - self.inventory_division - self.min_inventory) // self.inventory_division))

    def _discretize_day(self, day):
        return self._correct(int((day - self.day_division - self.min_day) // self.day_division))

    def _get_price(self, action):
        return (action + 1) * self.price_division + self.min_price

    def _build_state(self, observation):
        units1 = self._discretize_sales(observation[0])
        units2 = self._discretize_sales(observation[1])

        ranks = [self._discretize_rank(observation[i]) for i in range(2, 10)]

        inventory0 = self._discretize_inventory(observation[-4])
        adspend0 = self._discretize_adspend(observation[-3])

        day = self._discretize_day(int(observation[-2]) - 1)
        supply_status = int(observation[-1])

        state = [units1, units2] + ranks + [inventory0, adspend0, day, supply_status]
        res1, res2 = 1, 0
        for state, size in zip(state, self.sizes):
            res2 += state * res1
            res1 *= size

        return res2


# agent for training
class TabularQLearningAgent(TabularAgent):
    def __init__(self, min_price, max_price, bins_number, learning_rate=0.1, discount_factor=0.96,
                 exploration_rate=0.98, exploration_decay_rate=0.99):
        super().__init__(min_price, max_price, bins_number)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        prod = 1
        for size in self.sizes:
            prod *= size

        self.q = np.zeros((prod, bins_number), dtype=np.float32)

    def begin_episode(self, observation):
        self.state = self._build_state(observation)
        self.exploration_rate *= self.exploration_decay_rate

        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        self.action = np.random.randint(0, self.bins_number) if enable_exploration else self.q[self.state].argmax()

        return self._get_price(self.action)

    def act(self, observation, reward, done):
        if done:
            self.q[self.state][self.action] += self.learning_rate * \
                                               (reward - self.q[self.state][self.action])
            return

        next_state = self._build_state(observation)

        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        next_action = np.random.randint(0, self.bins_number) if enable_exploration else self.q[next_state].argmax()

        estimate = reward + self.discount_factor * self.q[next_state].max()
        self.q[self.state][self.action] += self.learning_rate * \
                                           (estimate - self.q[self.state][self.action])

        self.state = next_state
        self.action = next_action
        return self._get_price(self.action)

    def get_exploiting_agent(self):
        return ExploitingTabularAgent(self.min_price, self.max_price, self.bins_number, self.q)

    def register_reward(self, reward, episode):
        pass


# agent for exploiting learned action-values
class ExploitingTabularAgent(TabularAgent):
    def __init__(self, min_price, max_price, bins_number, q_table):
        super().__init__(min_price, max_price, bins_number)

        self.q = q_table

    @staticmethod
    def load(path):
        state_dict = pickle.load(open(path, "rb"))
        return ExploitingTabularAgent(**state_dict)

    def act(self, observation):
        state = self._build_state(observation)
        action = self.q[state].argmax()
        return self._get_price(action)

    def save(self, path):
        state_dict = {"min_price": self.min_price,
                      "max_price": self.max_price,
                      "bins_number": self.bins_number,
                      "q_table": self.q}
        pickle.dump(state_dict, open(os.path.join(path, "agent.pkl"), "wb"))

import numpy as np
from scipy import stats


# gym-compatible environment
class MarketEnv:
    def __init__(self, n_interactions, starting_states_df, demand_model, demand_residuals_parameters, adspend_model,
                 adspend_residuals_parameters, price_rank_coefficients, inventory_increase_parameters, normalize=True):
        self.n_interactions = n_interactions
        self.starting_states_df = starting_states_df
        self.demand_model = demand_model
        self.demand_residuals_parameters = demand_residuals_parameters
        self.adspend_model = adspend_model
        self.adspend_residuals_parameters = adspend_residuals_parameters
        self.price_to_rank_fn = np.poly1d(price_rank_coefficients)
        self.inventory_increase_probability = inventory_increase_parameters[0]
        self.inventory_increase_parameters = inventory_increase_parameters[1:]
        self.normalize = normalize

    @staticmethod
    def _normalize(state):
        sales_mean, sales_std = 124.4, 67.8
        rank_mean, rank_std = 38357.2, 33871.6
        inventory_mean, inventory_std = 9988.7, 8208.7
        adspend_mean, adspend_std = 101.5, 82.3
        day_mean, day_std = 15.5, 8.6
        inventory_increase_mean, inventory_increase_std = 0.2, 0.14

        mean_vector = np.array([sales_mean] * 2 + [rank_mean] * 8 + [inventory_mean, adspend_mean, day_mean,
                                                                     inventory_increase_mean])
        std_vector = np.array([sales_std] * 2 + [rank_std] * 8 + [inventory_std, adspend_std, day_std,
                                                                  inventory_increase_std])
        return (state - mean_vector) / std_vector

    def reset(self):
        self.count = 0
        starting_state = self.starting_states_df.sample().iloc[0]
        starting_state = starting_state.drop("rank_lag0")
        starting_state = starting_state.to_numpy()
        starting_state = np.concatenate((starting_state, np.array([0])))
        self.state = starting_state
        return self._normalize(self.state) if self.normalize else self.state

    def step(self, action):
        self.count += 1

        rank = self.price_to_rank_fn(action)
        demand_model_input = np.concatenate((self.state[:2], np.array([rank]), self.state[2:-2]))

        expected_sales = self.demand_model.predict(demand_model_input[None, :])[0]
        sales = expected_sales + stats.norm.rvs(loc=self.demand_residuals_parameters[0], scale=self.demand_residuals_parameters[1])
        sales = max(0, sales)

        expected_adspend = self.adspend_model.predict(np.array([[demand_model_input[-1]]]))[0]
        adspend = expected_adspend + stats.laplace.rvs(loc=self.adspend_residuals_parameters[0],
                                                       scale=self.adspend_residuals_parameters[1])
        adspend = max(0, adspend)

        prev_day = self.state[-2]
        day = prev_day % 30 + 1

        inventory = demand_model_input[-2] - sales
        increase_inventory = 0 if day == 1 else self.state[-1]
        if increase_inventory == 0 and np.random.rand() < self.inventory_increase_probability:
            increase_inventory = 1
            inventory = inventory + stats.expon.rvs(loc=self.inventory_increase_parameters[0],
                                                    scale=self.inventory_increase_parameters[1])

        self.state = np.concatenate((np.array([sales, demand_model_input[0]]), demand_model_input[2: -3],
                                     np.array([inventory, adspend, day, increase_inventory])))
        reward = sales * action
        done = self.count == self.n_interactions

        return self._normalize(self.state) if self.normalize else self.state, reward, done

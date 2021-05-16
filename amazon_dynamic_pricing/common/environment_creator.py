import pickle
import numpy as np
import pandas as pd

from amazon_dynamic_pricing.market_modeling.market_environment import MarketEnv


def create_environment(config, normalize=True):
    demand_model = pickle.load(open(config["demand_model_path"], "rb"))
    demand_residuals_parameters = np.load(config["demand_residuals_parameters_path"])
    adspend_model = pickle.load(open(config["adspend_model_path"], "rb"))
    adspend_residuals_parameters = np.load(config["adspend_residuals_parameters_path"])
    states_df = pd.read_csv(config["starting_states_path"])
    inventory_increase_parameters = np.load(config["inventory_increase_parameters_path"])
    price_rank_coefficients = np.load(config["price_rank_curve_coefficients_path"])

    return MarketEnv(config["n_interactions"], states_df, demand_model, demand_residuals_parameters, adspend_model,
                     adspend_residuals_parameters, price_rank_coefficients, inventory_increase_parameters, normalize)

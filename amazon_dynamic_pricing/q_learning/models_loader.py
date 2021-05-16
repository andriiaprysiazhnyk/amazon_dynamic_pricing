from dynamic_pricing.q_learning.tabular_learning.agents import ExploitingTabularAgent
from dynamic_pricing.q_learning.dqn_learning.agents import ExploitingDQNAgent


def load(path):
    if path.endswith(".pkl"):
        return ExploitingTabularAgent.load(path)
    elif path.endswith(".pth"):
        return ExploitingDQNAgent.load(path)

    raise TypeError("'path' argument must have extension .pkl or .pth")

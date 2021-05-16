from dynamic_pricing.simple_agents.agents import GreedyAgent, ConstantAgent, RandomAgent


def load(path):
    if path.endswith("random_agent.pkl"):
        return RandomAgent.load(path)
    elif path.endswith("min_agent.pkl") or path.endswith("max_agent.pkl"):
        return ConstantAgent.load(path)
    elif path.endswith("greedy_agent.pkl"):
        return GreedyAgent.load(path)

    raise TypeError("'path' argument must have extension .pkl or .pth")

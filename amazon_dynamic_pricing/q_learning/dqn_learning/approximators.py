import torch
from torch import nn
from collections import OrderedDict


class QNetwork(nn.Module):
    def __init__(self, fc_layers):
        super(QNetwork, self).__init__()

        self.fc_layers = fc_layers
        modules = [("layer{}".format(i), nn.Sequential(nn.Linear(fc_layers[i - 1], fc_layers[i]), nn.ReLU()))
                   for i in range(1, len(fc_layers) - 1)]

        self.model = nn.Sequential(OrderedDict(modules +
                                               [("layer{}".format(len(fc_layers)), nn.Linear(fc_layers[-2],
                                                                                             fc_layers[-1]))]))

    def get_optimal_action(self, features):
        with torch.no_grad():
            q_estimates = self(features.unsqueeze(0))
        return q_estimates.argmax().item()

    def get_action_value(self, features, action):
        with torch.no_grad():
            q_estimates = self(features.unsqueeze(0))
        return q_estimates[0, action].item()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)

    def get_state_dict(self):
        return {
            "fc_layers": self.fc_layers,
            "weights": self.state_dict()
        }


def get_network(state_dim, actions_number, layers):
    return QNetwork([state_dim] + layers + [actions_number])

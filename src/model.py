import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoplanetModel(nn.Module):
    def __init__(self, input_features=5, hidden_neurons=128):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons // 2)
        self.fc3 = nn.Linear(hidden_neurons // 2, 1)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
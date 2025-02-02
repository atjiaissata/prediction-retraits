import torch
import torch.nn as nn

import torch.nn as nn

class Regressor(nn.Module):
    
    def __init__(self, model_type, input_size, hidden_size, num_layers):
        super(Regressor, self).__init__()
        self.model_type = model_type

        if model_type == 'MLP':
            self.fc1 = nn.Linear(input_size, 30)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(30, 1)

        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        if self.model_type == 'MLP':
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x

        elif self.model_type == 'GRU':
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])
            return out

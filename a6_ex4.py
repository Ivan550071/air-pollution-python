import torch
import torch.nn as nn

class PM_Model(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout=0.5):
        super(PM_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x shape: (batch, input_size)
        return self.net(x)
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Layers
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # Activation function
        self.relu = nn.ReLU()
        # 20% dropout rate
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flattening input
        x = x.view(-1, 28 * 28)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

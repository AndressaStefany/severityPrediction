try:
    import torch.nn as nn
except Exception:
    pass

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out
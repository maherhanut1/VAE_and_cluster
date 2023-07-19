import torch
from torch import nn

class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':

    test_model = BasicModel()
    x = torch.normal(0, 1, (1, 3, 128,128))
    y = test_model(x)
    print(y.shape)
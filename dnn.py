import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(119, 119, 3), nn.BatchNorm2d(119), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(119, 119, 3), nn.BatchNorm2d(119))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu.forward(self.conv2.forward(self.conv1.forward(x)) + x)


class DNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(119, 119, 3), nn.BatchNorm2d(119), nn.ReLU(inplace=True)
        )
        self.tower = [ResBlock() for _ in range(20)]
        self.valueHead = nn.Sequential(nn.Conv2d(119, 1, 1),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU(inplace=True),
                                       nn.Flatten(),
                                       nn.Linear(7616, 256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 1),
                                       nn.Tanh())
        self.policyHead = nn.Sequential(nn.Conv2d(119, 2, 1),
                                        nn.BatchNorm2d(2),
                                        nn.ReLU(inplace=True),
                                        nn.Flatten(),
                                        nn.Linear(15232, 4672),
                                        nn.Softmax())
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(119, 119, 3)
        self.tower = nn.Res
"""

CNN model for wafer map classification.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferCNN(nn.Module):

    def __init__(self, n_classes=9, in_ch=2, widths=(32, 64, 128),
                 hidden=128, dropout=0.5, in_size=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.conv2 = nn.Conv2d(widths[0], widths[1], 3, padding=1)
        self.conv3 = nn.Conv2d(widths[1], widths[2], 3, padding=1)

        side = in_size // 8
        self.fc = nn.Linear(widths[2] * side * side, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   

        x = x.flatten(1)                             

        x = self.dropout(F.relu(self.fc(x)))
        return self.out(x)


def build_model(n_classes=9, in_ch=2, dropout=0.5, in_size=64):
    return WaferCNN(n_classes=n_classes, in_ch=in_ch, dropout=dropout,
                    in_size=in_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 2, 64, 64)
    print(f"parameters: {count_parameters(m):,}")
    print(f"output: {tuple(m(x).shape)}")

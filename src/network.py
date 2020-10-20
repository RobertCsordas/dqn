import torch
from typing import List


class AtariNet(torch.nn.Module):
    def __init__(self, img_shape: List[int], n_actions: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        out_size = self.net(torch.zeros([1]+list(img_shape), dtype=torch.float32)).numel()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(input).view(input.shape[0], -1))
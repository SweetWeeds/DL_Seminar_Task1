import numpy as np
import tqdm
import operators as ops

class VGG8:
    def __init__(self):
        self.net = [
            # Layer 1 (B, 1, 28, 28) -> (B, 32, 28, 28)
            ops.Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ops.ReLU(),

            # Layer2 (B, 32, 28, 28) -> (B, 64, 14, 14)
            ops.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ops.ReLU(),
            ops.MaxPooling(kernel_size=2, stride=2),
            
            # Layer 3 (B, 64, 14, 14) -> (B, 64, 14, 14)
            ops.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ops.ReLU(),

            # Layer 4 (B, 64, 14, 14) -> (B, 128, 7, 7)
            ops.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ops.ReLU(),
            ops.MaxPooling(kernel_size=2, stride=2),

            # Layer 5 (B, 128, 7, 7) -> (B, 256, 7, 7)
            ops.Conv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ops.ReLU(),

            # Layer 6 (B, 256, 7, 7) -> (B, 256, 7, 7)
            ops.Conv2D(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            ops.ReLU(),

            # Layer 7 (B, 256*7*7) -> (B, 256)
            ops.FullyConnected(in_features=256*7*7, out_features=256),
            ops.ReLU(),

            # Layer 8 (B, 256) -> (B, 10)
            ops.FullyConnected(in_features=256, out_features=10)
        ]

    def backprop(self, loss):
        dout = loss
        for i in range(len(self.net), -1, -1):
            dout = self.net[i].backward(dout)

    def forward(self, x: np.ndarray):
        for op in self.net:
            x = op.forward(x)
        return x

    def save(self, fileName: str):
        pass
import numpy as np
import tqdm
import operators as ops

class VGG8:
    def __init__(self, loss_func: function):
        self.net = [
            # Layer 1
            ops.Conv2D("Conv2D-L1"),
            ops.ReLU("ReLU-L1"),

            # Layer2
            ops.Conv2D("Conv2D-L2"),
            ops.ReLU("ReLU-L2"),
            ops.MaxPooling("MaxPool-L2"),
            
            # Layer 3
            ops.Conv2D("Conv2D-L3"),
            ops.ReLU("ReLU-L3"),

            # Layer 4
            ops.Conv2D("Conv2D-L4"),
            ops.ReLU("ReLU-L4"),
            ops.MaxPooling("MaxPool-L4"),

            # Layer 5
            ops.Conv2D("Conv2D-L5"),
            ops.ReLU("ReLU-L5"),

            # Layer 6
            ops.Conv2D("Conv2D-L6"),
            ops.ReLU("ReLU-L6"),
            ops.MaxPooling("MaxPool-L6"),

            # Layer 7
            ops.FullyConnected("FC-L7", S=256, R=128),
            ops.ReLU("ReLU-L7"),

            # Layer 8
            ops.FullyConnected("FC-L8", S=128, R=10)
        ]

    def train():
        pass

    def forward(self, x: np.ndarray):
        for op in self.net:
            x = op.forward(x)
        return x

    def save(self, fileName: str):
        pass
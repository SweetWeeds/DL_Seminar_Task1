import numpy as np
import tqdm
import operators as ops

class VGG8:
    def __init__(self, loss_func: function):
        self.net = [
            # Layer 1
            ops.Conv2D(),
            ops.MaxPooling(),
            
            # Layer 2
            ops.Conv2D(),
            ops.MaxPooling(),

            # Layer 3
            ops.Conv2D(),
            ops.MaxPooling(),

            # Layer 4
            ops.Conv2D(),
            ops.MaxPooling()
        ]
        pass

    def train():
        for i in tqdm()

    def forward(self, x: np.ndarray):
        for op in self.net:
            x = op.forward(x)
        return x
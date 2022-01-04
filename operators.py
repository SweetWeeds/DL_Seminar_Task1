import numpy as np
import basic_ops

""" 2-D Convolution """
class Conv2D(basic_ops.operation):
    # S: width, R: height, C: channel, M: number
    def __init__(self, S: int, R: int, C: int, M: int, stride: int):
        self.weights = np.zeros(shape=(S, R, C, M), dtype=np.float)
        self.stride = stride
        pass

    def forward(self, inp: np.ndarray):

        pass

""" Max Pooling """
class MaxPooling(basic_ops.operation):
    def __init__(self):
        pass

    def forward(self):
        pass

""" Average Pooling """
class AvgPooling(basic_ops.operation):
    def __init__(self):
        pass

    def forward(self):
        pass

""" Fully Connected """
class FullyConnected(basic_ops.operation):
    def __init__(self):
        pass

    def forward(self):
        pass

""" ReLU """
class ReLU(basic_ops.operation):
    def __init__(self):
        pass

    def forward(self):
        pass

""" Cross Entropy """
class CrossEtropy(basic_ops.operation):
    def __init__(self):
        pass

    def forward(self):
        pass



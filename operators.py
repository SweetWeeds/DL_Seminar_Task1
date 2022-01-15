import numpy as np
import basic_ops

# Hyper-Params
h = 1e-4

def numerical_grad(f: function, x: np.array) -> np.array:
    return

def loss_func() -> np.array:
    pass

""" 2D Convolution """
class Conv2D(basic_ops.Layer):
    # S: width, R: height, C: channel, M: number
    def __init__(self, layer_name: str, S: int, R: int, C: int, M: int, P: int, stride: int):
        super().__init__(layer_name=layer_name)
        self.S, self.R, self.C, self.M, self.P = S, R, C, M, P
        self.stride  = stride
        self.weights = np.random.rand(self.M, self.R, self.S, self.C).astype(np.float32)
        self.grads   = np.random.rand(self.M, self.R, self.S, self.C).astype(np.float32)  # Gradients

    def forward(self, inp: np.ndarray) -> np.array:
        H, W, C = inp.shape
        OUTPUT_W = (W - self.S + 2 * self.P) / self.stride + 1
        OUTPUT_H = (H - self.R + 2 * self.P) / self.stride + 1
        ret = np.zeros(OUTPUT_H, OUTPUT_W).astype(np.float32)
        if (C != self.C):
            print(f"[ERROR:{self.layer_name}] Channel is not matching. (Kernel Shape:{self.S, self.R, self.C, self.M}, Input Shape:{W, H, C})")
        
        for w in range(OUTPUT_W):
            for h in range(OUTPUT_H):
                ret[h][w] = (inp[w*self.stride:w*self.stride+self.S, :, :], self.weights)

        return ret

""" Max Pooling """
class MaxPooling(basic_ops.Layer):
    def __init__(self, layer_name:str):
        super().__init__(layer_name=layer_name)
        pass

    def forward(self):
        pass

""" Average Pooling """
class AvgPooling(basic_ops.Layer):
    def __init__(self, layer_name:str):
        super().__init__(layer_name=layer_name)
        pass

    def forward(self):
        pass

""" Fully Connected """
class FullyConnected(basic_ops.Layer):
    def __init__(self, layer_name: str, S: int, R: int):
        super().__init__(layer_name=layer_name)
        self.S, self.R = S, R
        self.weight = np.random.rand(self.S, self.R).astype(np.float32)
        self.bias   = np.random.rand(self.S, 1).astype(np.float32)

    def forward(self, inp: np.ndarray):
        H, W = inp.shape
        return np.dot(inp, self.weight) + self.bias

""" ReLU """
class ReLU(basic_ops.Layer):
    def __init__(self, layer_name:str):
        super().__init__(layer_name=layer_name)
        pass

    def forward(self):
        pass

""" Cross Entropy """
class CrossEtropy(basic_ops.Layer):
    def __init__(self, layer_name:str):
        super().__init__(layer_name=layer_name)
        pass

    def forward(self):
        pass



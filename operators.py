from math import floor
import random
import numpy as np

WEIGHT_INIT_STD = 0.01

"""
    Cross Entropy Error
"""
def CrossEntropyError(y:np.ndarray, t:np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


"""
    Softmax
"""
def Softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


"""
    Image to Column
"""
def im2col(input_data, filter_size, stride=1, pad=0):
    B, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_size)//stride + 1
    out_w = (W + 2*pad - filter_size)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((B, C, filter_size, filter_size, out_h, out_w))

    for y in range(filter_size):
        y_max = y + stride*out_h
        for x in range(filter_size):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B*out_h*out_w, -1)
    return col


"""
    Column to Image
"""
def col2im(col, input_shape, filter_size, stride=1, pad=0):
    B, C, H, W = input_shape
    out_h = (H + 2*pad - filter_size)//stride + 1
    out_w = (W + 2*pad - filter_size)//stride + 1
    col = col.reshape(B, out_h, out_w, C, filter_size, filter_size).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((B, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_size):
        y_max = y + stride*out_h
        for x in range(filter_size):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


""" 
    2D Convolution 
"""
class Conv2D:
    """
        in_channels: Input Activation's Channel
        out_channels: Output Activation's Channel
        kernel_size: Kernel's size
        padding: padding size
    """

    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, padding : int, stride : int, name: str=""):
        self.name = name
        # Params
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.padding        = padding
        self.stride             = stride
        self.x_shape = None

        # Weights
        self.W = WEIGHT_INIT_STD*np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
        self.dW = np.zeros_like(self.W)
        self.dW_v = np.zeros_like(self.dW)
        self.b = np.zeros(out_channels)
        self.db = np.zeros_like(self.b)
        self.db_v = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape    # B: Batch, C: Channel, H: Height, W: Width
        OUTPUT_H = 1 + int((H - self.kernel_size + 2 * self.padding) / self.stride)
        OUTPUT_W = 1 + int((W - self.kernel_size + 2 * self.padding) / self.stride)

        col = im2col(x, self.kernel_size, self.stride, self.padding)
        col_W = self.W.reshape(self.out_channels, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(B, OUTPUT_H, OUTPUT_W, -1).transpose(0, 3, 1, 2)  # Batch, Channel, Height, Width

        self.x_shape = x.shape
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.W.shape)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x_shape, self.kernel_size, self.stride, self.padding)

        return dx

    def update(self, lr: float, m: float=None) -> None:
        if (m != None):
            self.dW_v = m * self.dW_v - lr *self.dW
            self.W += self.dW_v
            self.db_v = m * self.db_v - lr *self.db
            self.b += self.db_v
        else:
            self.W -= self.dW*lr
            self.b -= self.db*lr

"""
    Max Pooling
"""
class MaxPooling:
    def __init__(self, kernel_size : int, stride : int, name : str=""):
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None

    # x : Input Feature (B, C, H, W)
    # out: Output Feature (B, C, OUT_H, OUT_W)
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape    # Batch, Channel, Height, Width
        OUT_H = int(floor((H-self.kernel_size)/self.stride+1))
        OUT_W = int(floor((W-self.kernel_size)/self.stride+1))
        
        col = im2col(x, self.kernel_size, self.stride, 0)
        col = col.reshape(-1, self.kernel_size*self.kernel_size)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(B, OUT_H, OUT_W, C).transpose(0, 3, 1, 2) # Batch, Channel, Height, Width

        self.x_shape = x.shape
        self.arg_max = arg_max

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.kernel_size*self.kernel_size
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        dcol = dmax.reshape(dmax.shape[0], dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x_shape, self.kernel_size, self.stride, 0)

        return dx

    def update(self, lr: float, m: float):
        return


"""
    Fully Connected
"""
class FullyConnected:
    def __init__(self, in_feature : int, out_feature : int, name: str=""):
        self.name = name
        self.W = WEIGHT_INIT_STD*np.random.rand(in_feature, out_feature)
        self.dW = np.zeros_like(self.W)
        self.dW_v = np.zeros_like(self.W)
        self.b = np.zeros(out_feature).T
        self.db = np.zeros_like(self.b)
        self.db_v = np.zeros_like(self.b)
        self.x = None
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.x_shape = x.shape
        ret = np.dot(x, self.W) + self.b
        return ret

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.x_shape)
        return dx

    def update(self, lr: float, m: float=None) -> None:
        if (m != None):
            self.dW_v = m * self.dW_v - lr *self.dW
            self.W += self.dW_v
            self.db_v = m * self.db_v - lr *self.db
            self.b += self.db_v
        else:
            self.W -= self.dW*lr
            self.b -= self.db*lr


"""
    ReLU
"""
class ReLU:
    def __init__(self, name: str=""):
        self.name = name
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.copy(x)
        self.mask = x<0
        out[self.mask] = 0
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout

        return dx
    
    def update(self, lr: float, m: float):
        return


class SoftmaxWithLoss:
    def __init__(self, name: str=""):
        self.name = name
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)


    def forward(self, x, t):
        self.t = t
        self.y = Softmax(x)
        self.loss = CrossEntropyError(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

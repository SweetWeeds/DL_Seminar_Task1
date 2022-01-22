from math import floor
import numpy as np

""" 
    2D Convolution 
"""
class Conv2D:
    # FN: Filter Number, C: Channel, FH: Filter Height, FW: Filter WIdth, P: Padding, S: Stride
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, padding : int):
        pass

    def im2col(self, input_data):
        """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
        Parameters
        ----------
        input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩
        
        Returns
        -------
        col : 2차원 배열
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.P - self.FH)//self.S + 1
        out_w = (W + 2*self.P - self.FW)//self.S + 1

        img = np.pad(input_data, [(0,0), (0,0), (self.P, self.P), (self.P, self.P)], 'constant')
        col = np.zeros((N, C, self.FH, self.FW, out_h, out_w))

        for y in range(self.FH):
            y_max = y + self.S*out_h
            for x in range(self.FW):
                x_max = x + self.S*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.S, x:x_max:self.S]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def forward(self, x: np.ndarray) -> np.array:
        super().forward()
        N, C, H, W = x.shape    # N: Batch, C: Channel, H: Height, W: Width
        OUTPUT_H = (int) ((H - self.FH + 2 * self.P) / self.S + 1)
        OUTPUT_W = (int) ((W - self.FW + 2 * self.P) / self.S + 1)

        col = self.im2col(x)
        kernel_W = self.kernel.reshape(self.FN, -1).T
        ret = np.dot(col, kernel_W) + self.b
        ret = ret.reshape(N, OUTPUT_H, OUTPUT_W, -1).transpose(0, 3, 1, 2)

        return ret


"""
    Max Pooling
"""
class MaxPooling:
    def __init__(self, kernel_size : int, stride : int):
        self.kernel_size = kernel_size
        self.stride = stride

    # x (B, C, H, W)
    def forward(self, x: np.ndarray):
        B, C, H, W = x.shape
        output = np.zeros((B, C, int(floor((H-self.kernel_size)/self.stride+1)), int(floor((W-self.kernel_size)/self.stride+1))), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                for h in range(0, H, self.stride):
                    for w in range(0, W, self.stride):
                        print(f"b:{b}, c:{c}, h:{h}, w:{w}")
                        print(x[b, c, h:h+self.stride, w:w+self.stride].shape)
                        output[b,c,h,w] = np.max(x[b, c, h:h+self.stride, w:w+self.stride])
        return output


"""
    Fully Connected
"""
class FullyConnected:
    def __init__(self, layer_name: str, InputSize: int, OutputSize: int):
        super().__init__(layer_name=layer_name)
        self.InputSize, self.OutputSize = InputSize, OutputSize

        self.weight = np.random.rand(self.InputSize, self.OutputSize).astype(np.float32)
        self.weight_grad = np.zeros_like(self.weight)

        self.bias = np.random.rand(self.OutputSize).astype(np.float32)
        self.bias_grad = np.zeros_like(self.bias)

    def backward(self, E: np.ndarray) -> np.ndarray:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward()
        ret = np.dot(x, self.weight) + self.bias
        return ret


"""
    ReLU
"""
class ReLU:
    def __init__(self, layer_name:str):
        super().__init__(layer_name=layer_name)

    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward()
        ret = np.copy(x)
        ret[ret<0] = 0
        return ret


"""
    SoftMax
"""
class SoftMax:
    def __init__(self, layer_name: str):
        super().__init__(layer_name=layer_name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward()
        c = np.max(x)
        exp_inp_c = np.exp(x - c)
        sum_exp_inp_c = np.sum(exp_inp_c)
        return exp_inp_c / sum_exp_inp_c


"""
    Loss Function: Cross Entropy Error
"""
def CrossEtropyError(y:np.ndarray, t:np.ndarray):
        delta = 1e-7
        return -np.sum(t*np.log(y+delta))

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class VGG8:
    """단순한 합성곱 신경망
    
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_size':3, 'pad':1, 'stride':1},
                 output_size=10, weight_init_std=0.01):
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

        # 가중치 초기화
        self.params = {}
        # (B, 1, 28, 28) -> (B, 32, 28, 28)
        self.params['L1_Conv_W'] = weight_init_std * \
                            np.random.randn(32, input_dim[0], filter_size, filter_size)
        self.params['L1_Conv_b'] = np.zeros(32)

        # (B, 32, 28, 28) -> (B, 64, 14, 14)
        self.params['L2_Conv_W'] = weight_init_std * \
                            np.random.randn(64, 32, filter_size, filter_size)
        self.params['L2_Conv_b'] = np.zeros(64)

        # (B, 64, 14, 14) -> (B, 64, 14, 14)
        self.params['L3_Conv_W'] = weight_init_std * \
                            np.random.randn(64, 64, filter_size, filter_size)
        self.params['L3_Conv_b'] = np.zeros(64)

        # (B,64, 14, 14) -> (B, 128, 7, 7)
        self.params['L4_Conv_W'] = weight_init_std * \
                            np.random.randn(128, 64, filter_size, filter_size)
        self.params['L4_Conv_b'] = np.zeros(128)

        # (B, 128, 7, 7) -> (B, 256, 7, 7)
        self.params['L5_Conv_W'] = weight_init_std * \
                            np.random.randn(256, 128, filter_size, filter_size)
        self.params['L5_Conv_b'] = np.zeros(256)

        # (B, 256, 7, 7) -> (B, 256, 7, 7)
        self.params['L6_Conv_W'] = weight_init_std * \
                            np.random.randn(256, 256, filter_size, filter_size)
        self.params['L6_Conv_b'] = np.zeros(256)

        # (B, 256*7*7) -> (B, 256)
        self.params['L7_Affine_W'] = weight_init_std * \
                            np.random.randn(256*7*7, 256)
        self.params['L7_Affine_b'] = np.zeros(256)

        # (B, 256) -> (B, 10)
        self.params['L8_Affine_W'] = weight_init_std * \
                            np.random.randn(256, 10)
        self.params['L8_Affine_b'] = np.zeros(10)


        # 계층 생성
        self.layers = OrderedDict()
        # Layer 1
        self.layers['L1_Conv'] = Convolution(self.params['L1_Conv_W'], self.params['L1_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L1_Relu'] = Relu()

        # Layer 2
        self.layers['L2_Conv'] = Convolution(self.params['L2_Conv_W'], self.params['L2_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L2_Relu'] = Relu()
        self.layers['L2_Pool'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # Layer 3
        self.layers['L3_Conv'] = Convolution(self.params['L3_Conv_W'], self.params['L3_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L3_Relu'] = Relu()

        # Layer 4
        self.layers['L4_Conv'] = Convolution(self.params['L4_Conv_W'], self.params['L4_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L4_Relu'] = Relu()
        self.layers['L4_Pool'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # Layer 5
        self.layers['L5_Conv'] = Convolution(self.params['L5_Conv_W'], self.params['L5_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L5_Relu'] = Relu()

        # Layer 6
        self.layers['L6_Conv'] = Convolution(self.params['L6_Conv_W'], self.params['L6_Conv_b'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['L6_Relu'] = Relu()

        # Layer 7
        self.layers['L7_Affine'] = Affine(self.params['L7_Affine_W'], self.params['L7_Affine_b'])
        self.layers['L7_Relu'] = Relu()

        # Layer 8
        self.layers['L8_Affine'] = Affine(self.params['L8_Affine_W'], self.params['L8_Affine_b'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer_name in self.layers.keys():
            try:
                x = self.layers[layer_name].forward(x)
            except:
                print(f"[{layer_name}]")
                exit()

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['L1_Conv_W'], grads['L1_Conv_b'] = self.layers['L1_Conv'].dW, self.layers['L1_Conv'].db
        grads['L2_Conv_W'], grads['L2_Conv_b'] = self.layers['L2_Conv'].dW, self.layers['L2_Conv'].db
        grads['L3_Conv_W'], grads['L3_Conv_b'] = self.layers['L3_Conv'].dW, self.layers['L3_Conv'].db
        grads['L4_Conv_W'], grads['L4_Conv_b'] = self.layers['L4_Conv'].dW, self.layers['L4_Conv'].db
        grads['L5_Conv_W'], grads['L5_Conv_b'] = self.layers['L5_Conv'].dW, self.layers['L5_Conv'].db
        grads['L6_Conv_W'], grads['L6_Conv_b'] = self.layers['L6_Conv'].dW, self.layers['L6_Conv'].db
        grads['L7_Affine_W'], grads['L7_Affine_b'] = self.layers['L7_Affine'].dW, self.layers['L7_Affine'].db
        grads['L8_Affine_W'], grads['L8_Affine_b'] = self.layers['L8_Affine'].dW, self.layers['L8_Affine'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val


        #for i, key in enumerate(['L1_Conv', 'Affine1', 'Affine2']):
        for i, key in enumerate(self.layers.keys):
            self.layers[key].W = self.params[key+'_W']
            self.layers[key].b = self.params[key+'_b']

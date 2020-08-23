import signal
import scipy.signal
from scipy.signal import *

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

"""
STFT_network의 입력 : [batch_size, time_step (signal length), channel == 1]
STFT_network의 출력 : return real, imag
_, signal = scipy.io.wavfile.read("./test_speech/speech.wav")
sound = np.reshape(signal, (1, -1, 1))
print(sound.shape)
(1, 81168, 1)

STFT_network의 입력 : [batch_size, time_step (signal length), channel == 1]
STFT_network의 출력 : return real, imag
real : [batch_size, time_step, frequency_bin]
imag : [batch_size, time_step, frequency_bin]
    
    네트워크의 출력 자체는 time_step과 frequency_bin이 transpose 상태
    (스펙토그램이 옆으로 누워있는 형상)
    이 출력의 스펙토그램을 보고 싶으면 다음과 같이 할 것
        spec = tf.transpose(tf.complex(real, imag), perm = [0, 2, 1])
        show_spectogram (spec, shape = (가로, 세로)

1. Convolution Network에 넣고 싶으면
    real, imag <== tf.reshape(real or imag, (batch_size, time_step, frequency_bin, 1)) 형태로 줄 것
    만약 [batch_size, time_step, frequency_bin]를 [batch_size, frequency_bin, time_step, 1] 처럼 주고 싶다면...
    real, imag <== tf.transpose(real or imag, perm = [0, 2, 1, 3]) <== tf.reshape(real or imag, (batch_size, time_step, frequency_bin, 1))


2. 중간 네트워크가 없다면
    ISTFT_network의 입력 : STFT_network의 출력을 그대로 이어서 받으면 됨
    ISTFT_network의 출력 : return signal [batch_size, time_step (signal length), channel]
    real : [batch_size, time_step, frequency_bin]
    imag : [batch_size, time_step, frequency_bin]

3. 기본 옵션
    length = 1024, over_lapping = 256, padding = "same"
    그리고 레이어의 trainable option은 "False"가 기본
    학습을 하면 안됨. DFT kernel이 깨짐
"""

class STFT_network (tf.keras.layers.Layer):
    

    def __init__ (self, length = 1024, over_lapping = 256, padding = "same"):
        
        super(STFT_network, self).__init__()
        
        self.window_length = length
        self.frequency_bin = int(self.window_length/2 + 1)
        self.over_lapping  = over_lapping
        self.padding = padding
        
        self.fourier_basis = np.fft.fft(np.eye(self.window_length))
        self.discrete_fourier_transform_window = scipy.signal.hann(self.window_length, sym = False)
        self.discrete_fourier_transform_window = self.discrete_fourier_transform_window.reshape((1, -1))
        
        self.kernel = np.multiply(self.fourier_basis, self.discrete_fourier_transform_window)
        del (self.fourier_basis)
        
        self.kernel = self.kernel[:self.frequency_bin, :]
        
        self.real_kernel_init = np.real(self.kernel)
        self.imag_kernel_init = np.imag(self.kernel)

        self.real_kernel_init = self.real_kernel_init.T
        self.imag_kernel_init = self.imag_kernel_init.T
        self.real_kernel_init = self.real_kernel_init[:, None, :]
        self.imag_kernel_init = self.imag_kernel_init[:, None, :]
        
        
    def build (self, inputs_shape):
        '''
        self.frequency_bin : integer
        self.window_length : integer
        self.strides : same as over_lapping number
        kernel_initializer : 3D Tensor [width = window_length, height = 1, output size : self.frequency_bin]
        '''  
        self.real_fourier_convolution = Conv1D(filters = self.frequency_bin, 
                                                kernel_size = self.window_length, 
                                                strides = self.over_lapping, 
                                                padding = self.padding,
                                                kernel_initializer = tf.keras.initializers.Constant(value = self.real_kernel_init),
                                                trainable = False)
        
        self.imag_foruier_convolution = Conv1D(filters = self.frequency_bin, 
                                                kernel_size = self.window_length, 
                                                strides = self.over_lapping, 
                                                padding = self.padding,
                                                kernel_initializer = tf.keras.initializers.Constant(value = self.imag_kernel_init),
                                                trainable = False)
        
        super(STFT_network, self).build(inputs_shape)
        

    def call (self, input_signal):
        '''
        inputs_signal : 3D Tensor [batch_size, signal_length, channel_number]
        '''
        real = self.real_fourier_convolution(input_signal)
        imag = self.imag_foruier_convolution(input_signal)
        
        return real, imag


    
class ISTFT_network (tf.keras.layers.Layer):
    

    def __init__ (self, length = 1024, over_lapping = 256, padding = "same"):
        
        super(ISTFT_network, self).__init__()
        
        self.length = length
        self.over_lapping = over_lapping
        
        self.cut_off = int(self.length / 2 + 1)
        self.kernel_size = length
        self.strides = over_lapping
        self.padding = padding
        
        """
        
        """
        self.window_coefficient = scipy.signal.get_window("hanning", self.length)
        self.inverse_window = self.inverse_stft_window(self.window_coefficient, self.over_lapping)

        # 역푸리에 변환 커널
        self.fourier_basis = np.fft.fft(np.eye(self.length))
        self.fourier_basis = np.vstack([np.real(self.fourier_basis[:self.cut_off, :]), 
                                        np.imag(self.fourier_basis[:self.cut_off, :])])

        self.inverse_basis = self.inverse_window * np.linalg.pinv(self.fourier_basis).T[:, None, None, ]
        self.inverse_basis = self.inverse_basis.T

        
    def inverse_stft_window (self, window, hop_length):

            # Ceiling division.
            window_length = len(window)
            denom = window ** 2
            overlaps = -(-window_length // hop_length) 
            denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
            denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
            denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)

            return window / denom[:window_length]
        
        
    def build (self, inputs_shape):
        
        self.expand_dims_lambda = Lambda(lambda x: K.expand_dims(x, axis=2))
        self.Conv2DTranspose = Conv2DTranspose(filters = 1, 
                                                kernel_size = (self.kernel_size, 1), 
                                                strides = (self.strides, 1), 
                                                padding = self.padding,
                                                kernel_initializer = tf.keras.initializers.Constant(self.inverse_basis),
                                                trainable = False)
        self.squeeze_dims_lambda = Lambda(lambda x: K.squeeze(x, axis=2))
        
        super(ISTFT_network, self).build(inputs_shape)
        
        
    def call (self, real, imag):
        '''
        inputs_signal : 3D Tensor [batch_size, time_step (signal length), channel_number (Frequency bin)]
        Inverse Short Time Fourier Transform을 하기 위해서는...
        입력 텐서의 크기가 배치 사이즈, 신호의 길이, 신호의 채널 형태로 들어와야 한다.
        Spectogram의 입력 형태사 [배치 사이즈, 타임 스텝, 주파수 bin] 이고

        이를 axis = 2에서 concat한 다음
        다시 axis = 2에서 expand_dims를 수행하면 [배치 사이즈. 타임 스텝, 1, 주파수 빈] 이다.

        다시 말해서 타임 스텝 x 1 형태의 이미지가 주파수 빈 채널만큼 있는 셈

        Conv 2D Tranpose를 하고 출력을 보면
        [배치 사이즈, 신호의 길이, 1, 1] 형태이고

        squeeze로 사이즈를 줄여서 [배치 사이즈, 신호의 길이, 1] 형태로 만든다.
        '''
        input_tensor = tf.concat([real, imag], axis = 2)
        outputs = self.expand_dims_lambda(input_tensor)
        outputs = self.Conv2DTranspose(outputs)
        outputs = self.squeeze_dims_lambda(outputs)
        
        return outputs
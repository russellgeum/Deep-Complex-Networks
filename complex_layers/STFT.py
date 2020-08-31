import signal
import scipy.signal
from scipy.signal import *

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

"""
STFT_network's INPUT : [batch_size, time_step (signal length), channel == 1]
STFT_network's OUTPUT : return real, imag
_, signal = scipy.io.wavfile.read("./test_speech/speech.wav")
sound = np.reshape(signal, (1, -1, 1))
print(sound.shape)
(1, 16384, 1)

real : [batch_size, time_step, frequency_bin]
imag : [batch_size, time_step, frequency_bin]
    
    STFT Network's outputs is transformation of time_step, frequency_bin transpose
    So,
    spec = tf.transpose(tf.complex(real, imag), perm = [0, 2, 1])
    show_spectogram (spec, shape = (width, height))

1. If you want to be part of the Convolution Network,
    real, imag  <= tf.reshape in real or image (batch_size, time_step, Frequency_bin, 1)) form
    If you want to give [batch_size, time_step, Frequency_bin] like [batch_size, Frequency_bin, time_step, 1]...
    real, imag  <= tf.transpose(real or imag, perm = [0, 2, 1, 3]) == tf.resape(real or imag, (batch_size, time_step, Frequency_bin, 1))


2. Without an intermediate network,
    Input of ISTFT_network: STFT_network can be continuously received.
    Output of ISTFT_network : return signal [batch_size, time_step (signal length), channel]
    real : [batch_size, time_step, Frequency_bin]
    imag : [batch_size, time_step, Frequency_bin]

3. Basic Options
    length = 1024, over_lapping = 256, padding = "same"
    And the layer's trainable option is based on "False".
    Do not learn. DFT kernel is broken
"""


'Short Time Fourier Transform via Neural Network'
class STFT_network (tf.keras.layers.Layer):
    

    def __init__ (self, window_length = 1024, over_lapping = 256, padding = "same"):
        
        super(STFT_network, self).__init__()
        
        self.window_length = window_length
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

        'inputs_signal : 3D Tensor [batch_size, signal_length, channel_number]'
        real = self.real_fourier_convolution(input_signal)
        imag = self.imag_foruier_convolution(input_signal)
        
        return real, imag


'Inverse Short Time Fourier Transform via Neural Network'    
class ISTFT_network (tf.keras.layers.Layer):
    

    def __init__ (self, window_length = 1024, over_lapping = 256, padding = "same"):
        
        super(ISTFT_network, self).__init__()
        
        self.window_length = window_length
        self.over_lapping  = over_lapping
        
        self.cut_off     = int(self.window_length / 2 + 1)
        self.kernel_size = window_length
        self.strides     = over_lapping
        self.padding     = padding
        
        self.window_coefficient = scipy.signal.get_window("hanning", self.window_length)
        self.inverse_window     = self.inverse_stft_window(self.window_coefficient, self.over_lapping)

        'Inverse Fourier Transform Kernel'
        self.fourier_basis = np.fft.fft(np.eye(self.window_length))
        self.fourier_basis = np.vstack([np.real(self.fourier_basis[:self.cut_off, :]), np.imag(self.fourier_basis[:self.cut_off, :])])

        self.inverse_basis = self.inverse_window * np.linalg.pinv(self.fourier_basis).T[ :, None, None, ]
        self.inverse_basis = self.inverse_basis.T

        
    def inverse_stft_window (self, window, hop_length):

            'Ceiling Division'
            window_length = len(window)
            denom = window ** 2
            overlaps = -(-window_length // hop_length) 
            denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
            denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
            denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)

            return window / denom[:window_length]
        
        
    def build (self, inputs_shape):
        
        self.expand_dims_lambda = Lambda(lambda x: K.expand_dims(x, axis = 2))
        self.Conv2DTranspose    = Conv2DTranspose(filters = 1, 
                                                kernel_size = (self.kernel_size, 1), 
                                                strides = (self.strides, 1), 
                                                padding = self.padding,
                                                kernel_initializer = tf.keras.initializers.Constant(self.inverse_basis),
                                                trainable = False)
        self.squeeze_dims_lambda = Lambda(lambda x: K.squeeze(x, axis = 2))
        
        super(ISTFT_network, self).build(inputs_shape)
        
        
    def call (self, real, imag):
        '''
        inputs_signal : 3D Tensor [batch_size, time_step (signal length), channel_number (Frequency bin)]
        To do Inverse Short Time Fourier Transform...
        The size of the input tensor shall come in the form of the batch size, the length of the signal, and the channel form of the signal.
        The input modalities of the Spectogram [deployment size, time step, frequency bin] are.

        Concat it at Axis = 2.
        If you perform expand_dims again at Axis = 2, [Deployment size. Time step, 1, frequency bin.

        In other words, an image in the form of Time Step x 1 is as good as a frequency bin channel.

        If you do Conv 2D Transpose and look at the output,
        [batch size, signal length, 1, 1] It's a form.

        Reduce size with skew to form [batch size, signal length, 1].
        '''
        input_tensor = tf.concat([real, imag], axis = 2)
        outputs = self.expand_dims_lambda(input_tensor)
        outputs = self.Conv2DTranspose(outputs)
        outputs = self.squeeze_dims_lambda(outputs)
        
        return outputs
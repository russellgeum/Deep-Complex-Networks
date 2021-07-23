# Deep Complex Networks  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
# About Deep Complex Networks
Most deep learning models update real parameters during training.  
But should it be real parameter? Can you think about deep learning model trained complex numbers?  
Deep learning in complex numbers has more expressive power than in the real numbers.  
Some papers introduces the complex neural network frameworks and some activation function.  
This repository gives you complex neural networks, activation functions and som signal processing layer.  
![activation](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/sample/activation.png)
![architecture](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/sample/architecture.png)
# Requirements  
```
scipy
numpy
librsoa == 0.7.2
pytorch >= 1.8.0
tensorflow == 2.2

2021 02 03 Issue  
텐서플로우 2.2보다 큰 버전은 complex_BatchNorm 연산이 되지 않습니다.
2021 07 23 Issues
텐서플로우 버전은 더 이상 개발하지 않습니다.
```
# Directory  
```
./tensorflow
  ./complex_layers
      __init__.py
      activation.py
          def CReLU
          def zReLU
          def modReLU
          ... ...
      layer.py
          class complex_Dense
          class complex_Conv2D
          class complex_Conv1D
          ... ...
  ./spectral_layers
      __init__.py
      STFT.py
          class STFT
          class InverseSTFT
./pytorch
  ./spectral_layers
      __init__.py
      STFT.py
          class STFT
          class InverseSTFT
```
# Usage (Tensorflow Example)
```
Ex 1, (real, imag) -> complex_conv2d -> complex_activation -> complex_batchnorm
from complex_layers.layer import *
from complex_layers.activation import *

real_inputs = tf.keras.Input(shaep = (64, 64, 1))
imag_inputs = tf.keras.Input(shape = (64, 64, 1))

real, imag = complex_Conv2D(**argments)(real_inputs, imag_inputs)
real, imag = CReLU(real, imag)
real, imag = complex_BatchNormalization2D(real, imag)


real_inputs = tf.keras.Input(shaep = (64, 64, 1))
imag_inputs = tf.keras.Input(shape = (64, 64, 1))
real, imag = complex_BatchNormalization2D(real, imag)


Ex 2, (real, imag) -> complex_batchnorm with model.summary()
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 64, 64, 1)]  0
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 64, 64, 1)]  0
__________________________________________________________________________________________________
tf_op_layer_concat (TensorFlowO [(None, 64, 64, 2)]  0           input_1[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
complex__batch_norm2d (complex_ (None, 64, 64, 2)    10          tf_op_layer_concat[0][0]
__________________________________________________________________________________________________
tf_op_layer_strided_slice (Tens [(None, 64, 64, 1)]  0           complex__batch_norm2d[0][0]
__________________________________________________________________________________________________
tf_op_layer_strided_slice_1 (Te [(None, 64, 64, 1)]  0           complex__batch_norm2d[0][0]
==================================================================================================
Total params: 10
Trainable params: 5
Non-trainable params: 5
```
# Will be...    
```
Signal Processing Layer
FFT
Wavelet
MFCC

pytorch
class compelx_Dense
class complex_Conv1d
class complex_Conv2d
class complex_Conv3d
class complex_Conv1dTraspose
class complex_Conv2dTraspose
class complex_Conv3dTraspose
class complex_LSTM
class complex_Attention
class complex_BatchNorm1D
class complex_BatchNorm2D

tensorflow
class complex_Conv3D
class complex_Conv3DTranspose
class complex_LSTM
class complex_attention

pytorch
class compelx_Dense
class complex_Conv1d
class complex_Conv2d
class complex_Conv3d
class complex_Conv1dTraspose
class complex_Conv2dTraspose
class complex_Conv3dTraspose
class complex_LSTM
class complex_Attention
```
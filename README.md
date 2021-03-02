# Deep Complex Networks  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
# About Deep Complex Networks
Models of ordinary deep learning update parameters in the real numbers.  
But should it be real number?  
Can you think about deep learning model in the field of complex numbers?  
Deep learning in complex numbers has more expressive power than in the real numbers.  
This paper introduces the neural network module in the field of multiple numbers.  
And This introduces some active functions as possible.  
In this repo, the solution networks and proposed activation functions are implemented.  
It then examines the performance of active functions in the complex numbers.  
But some complex neural network layer have mathematical issues.  
![activation](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/sample/activation.png)
![architecture](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/sample/architecture.png)
# Requirements  
TensorFlow == 2.2  
**2021 02 03 Issue  
TensorFlow 2.4.1 is not support Complex Batchnormalization module  (I will modify thie problem)**  
# Directory  
!!!  My module assumes to input the real parts and imagnary parts separately.  
```
./modyle
  ./complex_layers
      __init__.py
      activations.py
          def CReLU
          def zReLU
          def modReLU
          ... ...
      networks.py
          class complex_Dense
          class complex_Conv2D
          class conplex_Conv2DTranspose
          class complex_Conv1D
          class complex_Conv1dTrasnpose
          class complex_MaxPooling
      normalization.py
          class complex_NaiveBatchNormalization
          class complex_Dense_BatchNorm
          class complex_BatchNorm1D
          class complex_BatchNorm2D
          def complex_BatchNormalization
          def complex_BatchNormalization1D
          def complex_BatchNormalization2D
  ./spectral_layers
      __init__.py
      STFT.py
          class STFT_network
          class ISTFT_network
```
# Usage (Example)
```
Ex 1, (real, imag) -> complex_conv2d -> complex_activation -> complex_batchnorm
from complex_layers.networks import *
from complex_layers.activation import *
from complex_layers.normalization import *

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
# Will be developed later  
```
class complex_Conv3D
class complex_Conv3DTranspose
class complex_LSTM
class complex_Transformer
```

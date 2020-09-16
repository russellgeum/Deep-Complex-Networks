# Deep Complex Networks  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
#
# About Deep Complex Networks
1. Models of ordinary deep learning update parameters in the real numbers area.
2. But should it be real number? Can you think about deep learning model in the field of complex numbers?
3. Deep learning in complex numbers has more expressive power than in the real numbers. (It is said that there are many advantages.)
4. This paper introduces the neural network module in the field of multiple numbers, and introduces some active functions as possible.
In this repo, the solution networks and proposed activation functions are implemented.  
It then examines the performance of active functions in the complex numbers and examines the following mathematical issues.  
#
# Directory  
!!!  My module assumes to input the real parts and imagnary parts separately.  
```
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
        class complex_MaxPooling

    normalization.py
        class complex_NaiveBatchNormalization
        class complex_Batchnormalization2D
./spectral_layers
    __init__.py
    STFT.py
        class STFT_network
        class ISTFT_network
```
#
# Will be developed later  
```
class complex_Conv3D
class complex_Conv3DTranspose
class complex_LSTM
```
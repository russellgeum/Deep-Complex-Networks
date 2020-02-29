import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *

class complexConv2D ():
    """
    tf.keras.layers.Conv2D(filters, 
                            kernel_size, 
                            strides=(1, 1), 
                            padding='valid', 
                            data_format=None,
                            dilation_rate=(1, 1), 
                            activation=None, 
                            use_bias=True,
                            kernel_initializer='glorot_uniform', 
                            bias_initializer='zeros',
                            kernel_regularizer=None, 
                            bias_regularizer=None, 
                            activity_regularizer=None,
                            kernel_constraint=None, 
                            bias_constraint=None, 
                            **kwargs)
    """
    
    def __init__ (self, filters, 
                        kernel_size = (3, 3), 
                        strides = (2, 2), 
                        padding = "same", 
                        data_format=None,
                        dilation_rate=(1, 1), 
                        activation=None, 
                        use_bias=True,
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros',
                        kernel_regularizer=None, 
                        bias_regularizer=None, 
                        activity_regularizer=None,
                        kernel_constraint=None, 
                        bias_constraint=None):
        
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        
        self.realConv2D = Conv2D(filters, 
                                 kernel_size = self.kernel_size, 
                                 strides = self.strides,
                                 padding = self.padding,
                                 activation = self.activation)
                                 
        self.imagConv2D = Conv2D(filters, 
                                 kernel_size = self.kernel_size, 
                                 strides = self.strides,
                                 padding = self.padding,
                                 activation = self.activation)
        
        
    def fowardConvolution (self, real_inputs, imag_inputs):
    
        realOutputs = self.realConv2D (real_inputs) - self.imagConv2D  (imag_inputs)
        imagOutputs = self.imagConv2D (real_inputs) + self.realConv2D  (imag_inputs)
        
        output1 = realOutputs
        output2 = imagOutputs
        
        return output1, output2



class complexDense():
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """

    def __init__ (self, units, 
                        activation=None, 
                        use_bias=True, 
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', 
                        kernel_regularizer=None, 
                        bias_regularizer=None,
                        activity_regularizer=None, 
                        kernel_constraint=None, 
                        bias_constraint=None,
                        ):
        
        self.activation = activation
        
        self.realDense = Dense(units, activation = self.activation)
        self.imagDense = Dense(units, activation = self.activation)
        

    def fowardDense(self, real_inputs, imag_inputs):
        
        realOutputs = self.realDense (real_inputs) - self.imagDense  (imag_inputs)
        imagOutputs = self.imagDense (real_inputs) + self.realDense  (imag_inputs)
        
        output1 = realOutputs
        output2 = imagOutputs
        
        return output1, output2
import tensorflow as tf

def Conv2D (real_inputs, imag_inputs, filters, kernel_size):
    
    class complexConv2D ():

        def __init__ (self,
                    strides = (1, 1), 
                    padding = "same", 
                    activation = None, 
                    use_bias = True,
                    kernel_initializer = 'glorot_uniform', 
                    bias_initializer = 'zeros'):

            self.filters = filters
            self.kernel_size = kernel_size
            self.strides     = strides
            self.padding     = padding
            self.activation  = activation
            self.use_bias    = use_bias
            self.kernel_initializer = kernel_initializer
            self.bias_initializer   = bias_initializer

            self.realConv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                    kernel_size = self.kernel_size, 
                                                    strides = self.strides,
                                                    padding = self.padding,
                                                    activation = self.activation,
                                                    use_bias = self.use_bias,
                                                    kernel_initializer = self.kernel_initializer,
                                                    bias_initializer = self.bias_initializer) 

            self.imagConv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                    kernel_size = self.kernel_size, 
                                                    strides = self.strides,
                                                    padding = self.padding,
                                                    activation = self.activation,
                                                    use_bias = self.use_bias,
                                                    kernel_initializer = self.kernel_initializer,
                                                    bias_initializer = self.bias_initializer) 


        def fowardConvolution (self, real_inputs, imag_inputs):

            real_output = self.realConv2D(real_inputs) - self.imagConv2D(imag_inputs)
            imag_output = self.imagConv2D(real_inputs) + self.realConv2D(imag_inputs)

            output1 = real_output
            output2 = imag_output

            return output1, output2
    
    Conv2D = complexConv2D()
    
    real_output, imag_output = Conv2D.fowardConvolution(real_inputs, imag_inputs)
    
    output1 = real_output
    output2 = imag_output
    
    return output1, output2


def Conv2DTranspose (real_inputs, imag_inputs, filters, kernel_size):
    
    class complexConv2DTranspose ():

        def __init__(self,
                    strides = (2, 2), 
                    padding = 'same', 
                    activation = None, 
                    use_bias = True,
                    kernel_initializer = 'glorot_uniform', 
                    bias_initializer = 'zeros'):

            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.padding = padding
            self.activation = activation 
            self.use_bias = use_bias
            self.kernel_initializer = kernel_initializer
            self.bias_initializer = bias_initializer

            self.realConv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                            kernel_size = self.kernel_size, 
                                                            strides = self.strides, 
                                                            padding = self.padding, 
                                                            activation = self.activation, 
                                                            use_bias = self.use_bias,
                                                            kernel_initializer = self.kernel_initializer, 
                                                            bias_initializer = self.bias_initializer)

            self.imagConv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                            kernel_size = self.kernel_size, 
                                                            strides = self.strides, 
                                                            padding = self.padding, 
                                                            activation = self.activation, 
                                                            use_bias = self.use_bias,
                                                            kernel_initializer = self.kernel_initializer, 
                                                            bias_initializer = self.bias_initializer)

        def fowardConv2DTranspose (self, real_inputs, imag_inputs):

            real_outputs = self.realConv2DTranspose(real_inputs) - self.imagConv2DTranspose(imag_inputs)
            imag_outputs = self.imagConv2DTranspose(real_inputs) + self.realConv2DTranspose(imag_inputs)

            output1 = real_outputs
            output2 = imag_outputs

            return output1, output2
        
    Conv2DTranspose = complexConv2DTranspose()
    
    real_output, imag_output = Conv2DTranspose.fowardConv2DTranspose(real_inputs, imag_inputs)
    
    output1 = real_output
    output2 = imag_output
    
    return output1, output2

    
    
def Dense (real_inputs, imag_inputs, units):
    
    class complexDense():
        """
        tf.keras.layers.Dense(
            units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
            **kwargs
        )
        """

        def __init__ (self,
                    use_bias=True, 
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', 
                    kernel_regularizer=None, 
                    bias_regularizer=None,
                    activity_regularizer=None, 
                    kernel_constraint=None, 
                    bias_constraint=None):

            self.units = units
            self.realDense = tf.keras.layers.Dense(self.units, activation = self.activation)
            self.imagDense = tf.keras.layers.Dense(self.units, activation = self.activation)


        def fowardDense(self, real_inputs, imag_inputs):

            realOutputs = self.realDense(real_inputs) - self.imagDense(imag_inputs)
            imagOutputs = self.imagDense(real_inputs) + self.realDense(imag_inputs)

            output1 = realOutputs
            output2 = imagOutputs

            return output1, output2
        
    Dense = complexDense()
    
    real_output, imag_output = Dense.fowardDense(real_inputs, imag_inputs)
    
    output1 = real_output
    output2 = imag_output
    
    return output1, output2



def Pooling (real_inputs, imag_inputs, pool_size, padding = "same"):

    real_output = tf.keras.layers.MaxPool2D(pool_size = pool_size, 
                                            strides = None, 
                                            padding = padding) (real_inputs)
    imag_output = tf.keras.layers.MaxPool2D(pool_size = pool_size, 
                                            strides = None, 
                                            padding = padding) (imag_inputs)

    output1 = real_output
    output2 = imag_output
    
    return output1, output2
        
    
        
def BatchNomalization (self, 
                    real_inputs, 
                    imag_inputs,
                    momentum,
                    epsilon,
                    center,
                    scale,
                    beta_initializer,
                    gamma_initializer,
                    moving_mean_initializer,
                    moving_variance_initializer):

    realOutputs = tf.keras.layers.BatchNormalization(momentum = momentum, 
                            epsilon = epsilon, 
                            center = center, 
                            scale = scale,
                            beta_initializer = beta_initializer, 
                            gamma_initializer = gamma_initializer,
                            moving_mean_initializer = moving_mean_initializer,
                            moving_variance_initializer = moving_variance_initializer)(real_inputs)
    imagOutputs = tf.keras.layers.BatchNormalization(momentum = momentum, 
                            epsilon = epsilon, 
                            center = center, 
                            scale = scale,
                            beta_initializer = beta_initializer, 
                            gamma_initializer = gamma_initializer,
                            moving_mean_initializer = moving_mean_initializer,
                            moving_variance_initializer = moving_variance_initializer)(imag_inputs)

    output1 = realOutputs
    output2 = imagOutputs

    return output1, output2

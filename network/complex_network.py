from module import *


# Complex Convolution
##################################################################################################
class complex_Conv2D ():

    def __init__ (self, filters = 32,
                        kernel_size = (3, 3), 
                        strides = (2, 2), 
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

        self.real_conv2d = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 

        self.imag_conv2d = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 


    def foward (self, real_inputs, imag_inputs):

        real_outputs = self.real_conv2d(real_inputs) - self.imag_conv2d(imag_inputs)
        imag_outputs = self.imag_conv2d(real_inputs) + self.real_conv2d(imag_inputs)

        return real_outputs, imag_outputs


# Complex Transpose Conovolution
##################################################################################################
class conplex_Conv2DTranspose ():

    def __init__(self,  filters = 32,
                        kernel_size = (3, 3), 
                        strides = (2, 2), 
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

        self.real_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                        kernel_size = self.kernel_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding, 
                                                        activation = self.activation, 
                                                        use_bias = self.use_bias,
                                                        kernel_initializer = self.kernel_initializer, 
                                                        bias_initializer = self.bias_initializer)

        self.imag_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                        kernel_size = self.kernel_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding, 
                                                        activation = self.activation, 
                                                        use_bias = self.use_bias,
                                                        kernel_initializer = self.kernel_initializer, 
                                                        bias_initializer = self.bias_initializer)

    def foward (self, real_inputs, imag_inputs):

        real_outputs = self.real_Conv2DTranspose(real_inputs) - self.imag_Conv2DTranspose(imag_inputs)
        imag_outputs = self.imag_Conv2DTranspose(real_inputs) + self.real_Conv2DTranspose(imag_inputs)

        return real_outputs, imag_outputs

    
# Complex Dense
##################################################################################################
class complex_Dense():
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """
    def __init__ (self, units = 62,
                        activation = None,
                        use_bias=True, 
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', 
                        kernel_regularizer=None, 
                        bias_regularizer=None,
                        activity_regularizer=None, 
                        kernel_constraint=None, 
                        bias_constraint=None):

        self.units = units
        self.activation = activation
        self.real_dense = tf.keras.layers.Dense(self.units, activation = self.activation)
        self.imag_dense = tf.keras.layers.Dense(self.units, activation = self.activation)


    def foward (self, real_inputs, imag_inputs):

        real_outputs = self.real_dense(real_inputs) - self.imag_dense(imag_inputs)
        imag_outputs = self.imag_dense(real_inputs) + self.real_dense(imag_inputs)

        return real_outputs, imag_outputs



# Complex Pooling 
##################################################################################################
def MaxPooling (real_inputs, 
             imag_inputs, 
             pool_size, 
             strides = None, 
             padding = "same"):

    real_outputs = tf.keras.layers.MaxPool2D(pool_size = pool_size, 
                                            strides = strides, 
                                            padding = padding)(real_inputs)
    imag_outputs = tf.keras.layers.MaxPool2D(pool_size = pool_size, 
                                            strides = strides, 
                                            padding = padding)(imag_inputs)
    
    return real_outputs, imag_outputs
        
    
    
# Complex BatchNOmalization
##################################################################################################
def BatchNomalization (real_inputs, imag_inputs,
                        momentum, epsilon, center, scale,
                        beta_initializer, gamma_initializer,
                        moving_mean_initializer, moving_variance_initializer):

    real_outputs = tf.keras.layers.BatchNormalization(momentum = momentum, 
                                            epsilon = epsilon, 
                                            center = center, 
                                            scale = scale,
                                            beta_initializer = beta_initializer, 
                                            gamma_initializer = gamma_initializer,
                                            moving_mean_initializer = moving_mean_initializer,
                                            moving_variance_initializer = moving_variance_initializer)(real_inputs)
    imag_outputs = tf.keras.layers.BatchNormalization(momentum = momentum, 
                                            epsilon = epsilon, 
                                            center = center, 
                                            scale = scale,
                                            beta_initializer = beta_initializer, 
                                            gamma_initializer = gamma_initializer,
                                            moving_mean_initializer = moving_mean_initializer,
                                            moving_variance_initializer = moving_variance_initializer)(imag_inputs)

    return real_outputs, imag_outputs

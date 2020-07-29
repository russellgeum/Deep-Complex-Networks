import tensorflow as tf



# Complex Dense
##################################################################################################
class complex_Dense(tf.keras.layers.Layer):
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """
    def __init__ (self, units = 512,
                        activation = None,
                        use_bias   = True, 
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer   = 'zeros', 
                        kernel_regularizer = None, 
                        bias_regularizer   = None,
                        activity_regularizer = None, 
                        kernel_constraint    = None, 
                        bias_constraint      = None):
        
        super(complex_Dense, self).__init__()

        self.units = units
        self.activation = activation
        self.use_bias   = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer   = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint    = kernel_constraint
        self.bias_constraint      = bias_constraint
        
        
    def build (self, inputs_shape):
        
        self.real_Dense = tf.keras.layers.Dense(units = self.units, 
                                                activation = self.activation, 
                                                use_bias   = self.use_bias, 
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer   = self.bias_initializer, 
                                                kernel_regularizer = self.kernel_regularizer,
                                                bias_regularizer   = self.bias_regularizer,
                                                activity_regularizer = self.activity_regularizer, 
                                                kernel_constraint    = self.kernel_constraint,
                                                bias_constraint      = self.bias_constraint)
        
        self.imag_Dense = tf.keras.layers.Dense(units = self.units, 
                                                activation = self.activation, 
                                                use_bias   = self.use_bias, 
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer   = self.bias_initializer, 
                                                kernel_regularizer = self.kernel_regularizer,
                                                bias_regularizer   = self.bias_regularizer,
                                                activity_regularizer = self.activity_regularizer, 
                                                kernel_constraint    = self.kernel_constraint,
                                                bias_constraint      = self.bias_constraint)
        
        super(complex_Dense, self).build(inputs_shape)
        

    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Dense(real_inputs) - self.imag_Dense(imag_inputs)
        imag_outputs = self.imag_Dense(real_inputs) + self.real_Dense(imag_inputs)

        return real_outputs, imag_outputs



# Complex Convolution
##################################################################################################
class complex_Conv2D (tf.keras.layers.Layer):
    
    def __init__(self, 
                filters = 32,
                kernel_size = (3, 3), 
                strides = (2, 2), 
                padding = "same",
                activation = None,
                use_bias   = True,
                kernel_initializer = 'glorot_uniform',
                bias_initializer   = 'zeros'):
        
        super(complex_Conv2D, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.use_bias    = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        
        
    def build (self, inputs_shape):
        
        self.real_Conv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 

        self.imag_Conv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 
        
        super(complex_Conv2D, self).build(inputs_shape)

        
    def call(self, real_inputs, imag_inputs):
        
        real_outputs = self.real_Conv2D(real_inputs) - self.imag_Conv2D(imag_inputs)
        imag_outputs = self.imag_Conv2D(real_inputs) + self.real_Conv2D(imag_inputs)
        
        return real_outputs, imag_outputs



# Complex Transpose Conovolution
##################################################################################################
class conplex_Conv2DTranspose (tf.keras.layers.Layer):


    def __init__(self,  filters = 32,
                        kernel_size = (3, 3), 
                        strides = (2, 2), 
                        padding = "same",
                        activation = None,
                        use_bias   = True,
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer   = 'zeros'):
        
        super(conplex_Conv2DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.use_bias    = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        
        
    def build (self, inputs_shape):

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
        
        super(conplex_Conv2DTranspose, self).build(inputs_shape)

        
    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Conv2DTranspose(real_inputs) - self.imag_Conv2DTranspose(imag_inputs)
        imag_outputs = self.imag_Conv2DTranspose(real_inputs) + self.real_Conv2DTranspose(imag_inputs)

        return real_outputs, imag_outputs



# Complex Pooling 
##################################################################################################
class complex_MaxPooling (tf.keras.layers.Layer):

    def __init__(self,
                pool_size = (2, 2), 
                strides   = (1, 1),
                padding   = "same"):

        super(complex_MaxPooling, self).__init__()

        self.pool_size = pool_size
        self.strides   = strides
        self.padding   = padding


    def build (self, inputs_shape):
        
        self.real_maxpooling = tf.keras.layers.MaxPool2D(pool_size = self.pool_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding)
        
        self.imag_maxpooling = tf.keras.layers.MaxPool2D(pool_size = self.pool_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding)
        
        super(complex_MaxPooling, self).build(inputs_shape)
        

    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_maxpooling(real_inputs)
        imag_outputs = self.imag_maxpooling(imag_inputs)

        return real_outputs, imag_outputs



def complex_NaiveBatchNormalization (real_inputs, imag_inputs):

    real_outputs = tf.keras.layers.BatchNormalization()(real_inputs)
    imag_outputs = tf.keras.layers.BatchNormalization()(imag_inputs)

    return real_outputs, imag_outputs


# # Complex BatchNOmalization
# ##################################################################################################
# class complex_NaiveBatchNormalization (tf.keras.layers.Layer):

#     def __init__ (self):

#         super(complex_NaiveBatchNormalization, self).__init__()


#     def build (self, inputs_shape):

#         self.real_batchnorm = tf.keras.layers.BatchNormalization()
#         self.imag_batchnorm = tf.keras.layers.BatchNormalization()

#         super(complex_NaiveBatchNormalization, self).build(inputs_shape)

#     def call (self, real_inputs, imag_inputs):

#         real_outputs = self.real_batchnorm(real_inputs)
#         imag_outputs = self.imag_batchnorm(imag_inputs)

#         return real_outputs, imag_outputs

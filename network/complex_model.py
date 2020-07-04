from module import *
from complex_network import *
from complex_activation import *

class vgg16 ():

    def __init__ (self,
                kernel_size = (3, 3), 
                strides = (1, 1), 
                pool_size = (2, 2),
                node = 10,
                activation = "CReLU"):

        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.activation = activation
        
        self.node = node

    def activation_layer (self, real, imag):

        if self.activation is "CReLU":
            real, imag = CReLU(real, imag)
        elif self.activation is "zReLU":
            real, imag = zReLU(real, imag)
        elif self.activation is "modReLU":
            real, imag = modReLU(real, imag)

        return real, imag
    
    def convolution_layer (self, real, imag, filters):
        
        real, imag = complex_Conv2D(filters = filters, 
                                kernel_size = self.kernel_size, 
                                strides = self.strides).foward(real, imag)
        real, imag = self.activation_layer(real, imag)
        
        return real, imag


    def model (self, input_size = (224, 224, 3)):

        real_inputs = Input(shape = input_size)
        imag_inputs = Input(shape = input_size)
        
        real, imag = self.convolution_layer(real_inputs, imag_inputs, 64)
        real, imag = self.convolution_layer(real, imag, 64)
        real, imag = MaxPooling(real, imag, pool_size = self.pool_size)
        
        real, imag = self.convolution_layer(real, imag, 128)
        real, imag = self.convolution_layer(real, imag, 128)
        real, imag = MaxPooling(real, imag, pool_size = self.pool_size)
        
        real, imag = self.convolution_layer(real, imag, 256)
        real, imag = self.convolution_layer(real, imag, 256)
        real, imag = self.convolution_layer(real, imag, 256)
        real, imag = MaxPooling(real, imag, pool_size = self.pool_size)
        
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = MaxPooling(real, imag, pool_size = self.pool_size)
        
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = self.convolution_layer(real, imag, 512)
        real, imag = MaxPooling(real, imag, pool_size = self.pool_size)

        flatten_real, flatten_imag = flatten(real, imag)

        flat_real, flat_imag = complex_Dense(4096).foward(flatten_real, flatten_imag)
        flat_real, flat_imag = self.activation_layer(flat_real, flat_imag)
        flat_real, flat_imag = complex_Dense(1000).foward(flat_real, flat_imag)
        flat_real, flat_imag = self.activation_layer(flat_real, flat_imag)

        flat_real, flat_imag = complex_Dense(self.node).foward(flat_real, flat_imag)
        magnitude_outputs = softmax(flat_real, flat_imag)
        
        model = Model(inputs = [real_inputs, imag_inputs], outputs = magnitude_outputs)
        model.summary()

        return model

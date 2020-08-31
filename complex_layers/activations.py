import numpy as np
import tensorflow as tf


def complex_flatten (real, imag):
    
    real = tf.keras.layers.Flatten()(real)
    imag = tf.keras.layers.Flatten()(imag)
    
    return real, imag


def CReLU (real, imag):
    
    real = tf.keras.layers.ReLU()(real)
    imag = tf.keras.layers.ReLU()(imag)
    
    return real, imag


def zReLU (real, imag):

    real = tf.keras.layers.ReLU()(real)
    imag = tf.keras.layers.ReLU()(imag)
    
    real_flag = tf.cast(tf.cast(real, tf.bool), tf.float32)
    imag_flag = tf.cast(tf.cast(imag, tf.bool), tf.float32)
    
    flag = real_flag * imag_flag

    real = tf.math.multiply(real, flag)
    imag = tf.math.multiply(imag, flag)

    return real, imag


def modReLU (real, imag):
    
    norm = tf.abs(tf.complex(real, imag))
    bias = tf.Variable(np.zeros([norm.get_shape()[-1]]), trainable = True, dtype=tf.float32)
    relu = tf.nn.relu(norm + bias)
    
    real = tf.math.multiply(relu / norm + (1e+5), real)
    imag = tf.math.multiply(relu / norm + (1e+5), imag)
    
    return real, imag


def CLeaky_ReLU (real, imag):

    real = tf.nn.leaky_relu(real)
    imag = tf.nn.leaky_relu(imag)

    return real, imag


def complex_tanh (real, imag):

    real = tf.nn.tanh(real)
    imag = tf.nn.tanh(imag)

    return real, imag


def complex_softmax (real, imag):
    
    magnitude = tf.abs(tf.complex(real, imag))
    magnitude = tf.keras.layers.Softmax()(magnitude)
    
    return magnitude
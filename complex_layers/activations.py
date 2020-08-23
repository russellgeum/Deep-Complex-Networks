import numpy as np
import tensorflow as tf


def complex_flatten (input1, input2):
    
    output1 = tf.keras.layers.Flatten()(input1)
    output2 = tf.keras.layers.Flatten()(input2)
    
    return output1, output2


def CReLU (input1, input2):
    
    output1 = tf.keras.layers.ReLU()(input1)
    output2 = tf.keras.layers.ReLU()(input2)
    
    return output1, output2


def zReLU (input1, input2):

    real_relu = tf.keras.layers.ReLU()(input1)
    imag_relu = tf.keras.layers.ReLU()(input2)
    
    # 각 parts를 값이 있으면 True == 1로 만들고, 값이 0이면 False == 0을 반환
    real_flag = tf.cast(tf.cast(real_relu, tf.bool), tf.float32)
    imag_flag = tf.cast(tf.cast(imag_relu, tf.bool), tf.float32)
    
    # 각 part가 True == 1이면 1 반환, 하나라도 False == 0이면 0반환
    # 그래서 real, imag 중 하나라도 축 위에 값이 있으면 flag는 (0, ...) 이다.
    flag = real_flag * imag_flag

    # flag과 행렬끼리 원소곱을 하여, flag (1, ...)에서는 ReLU를 유지
    # (0, ...) flag에서는 값을 기각한다.
    real_relu = tf.math.multiply(real_relu, flag)
    imag_relu = tf.math.multiply(imag_relu, flag)

    return real_relu, imag_relu


def modReLU (input1, input2):
    
    norm = tf.abs(tf.complex(input1, input2))
    bias = tf.Variable(np.zeros([norm.get_shape()[-1]]), trainable = True, dtype=tf.float32)
    relu = tf.nn.relu(norm + bias)
    
    real = tf.math.multiply(relu / norm + (1e+5), input1)
    imag = tf.math.multiply(relu / norm + (1e+5), input2)
    
    return real, imag


def CLeaky_ReLU (input1, input2):

    output1 = tf.nn.leaky_relu(input1)
    output2 = tf.nn.leaky_relu(input2)

    return output1, output2


def complex_tanh (input1, input2):

    output1 = tf.nn.tanh(input1)
    output2 = tf.nn.tanh(input2)

    return output1, output2


def complex_softmax (input1, input2):
    
    magnitude = tf.abs(tf.complex(input1, input2))
    output = tf.keras.layers.Softmax()(magnitude)
    
    return output
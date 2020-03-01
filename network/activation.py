import tensorflow as tf


def Flatten (input1, input2):
    
    output1 = tf.keras.layers.Flatten()(input1)
    output2 = tf.keras.layers.Flatten()(input2)
    
    return output1, output2


def CReLU (input1, input2):
    
    output1 = tf.keras.layers.ReLU()(input1)
    output2 = tf.keras.layers.ReLU()(input2)
    
    return output1, output2


def zReLU (input1, input2):

    realReLU = tf.keras.layers.ReLU()(input1)
    imagReLU = tf.keras.layers.ReLU()(input2)
    
    # 각 parts를 값이 있으면 True == 1로 만들고, 값이 0이면 False == 0을 반환
    realFlag = tf.cast(tf.cast(realReLU, tf.bool), tf.float32)
    imagFlag = tf.cast(tf.cast(imagReLU, tf.bool), tf.float32)
    
    # 각 part가 True == 1이면 1 반환, 하나라도 False == 0이면 0반환
    # 그래서 real, imag 중 하나라도 축 위에 값이 있으면 flag는 (0, ...) 이다.
    flag = realFlag * imagFlag

    # flag과 행렬끼리 원소곱을 하여, flag (1, ...)에서는 ReLU를 유지
    # (0, ...) flag에서는 값을 기각한다.
    realReLU = tf.math.multiply(realReLU, flag)
    imagReLU = tf.math.multiply(imagReLU, flag)

    output1 = realReLU
    output2 = imagReLU

    return output1, output2


def Softmax (input1, input2):
    
    output1 = tf.keras.layers.Softmax()(input1)
    output2 = tf.keras.layers.Softmax()(input2)
    
    return output1, output2

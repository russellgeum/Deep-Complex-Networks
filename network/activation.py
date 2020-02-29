import tensorflow as tf

def Flatten (input1, input2):
    
    output1 = tf.keras.layers.Flatten()(input1)
    output2 = tf.keras.layers.Flatten()(input2)
    
    return output1, output2

def CReLU (input1, input2):
    
    output1 = tf.keras.layers.ReLU()(input1)
    output2 = tf.keras.layers.ReLU()(input2)
    
    return output1, output2

def Softmax (input1, input2):
    
    output1 = tf.keras.layers.Softmax()(input1)
    output2 = tf.keras.layers.Softmax()(input2)
    
    return output1, output2
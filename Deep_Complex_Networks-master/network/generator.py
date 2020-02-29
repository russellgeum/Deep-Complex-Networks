import numpy as np
import tensorflow as tf


"""
datagenerator for mnist images
"""
class dataGenerator():
    
    def __init__ (self):
        
        self.mnist = tf.keras.datasets.mnist
        (self.trainImage, self.trainLabel), (self.testImage, self.testLabel) = self.mnist.load_data()
        
    def NORMproccseing (self):
        
        self.trainImage = self.trainImage/255.
        self.testImage = self.testImage/255.
        
        return self.trainImage, self.testImage
    
    
    def FFTprocessing (self):
        
        trainImage, testImage = self.NORMproccseing()
        
        FFT_trainImage = np.fft.fft2(trainImage)
        FFT_testImage = np.fft.fft2(testImage)
        
        # real parts and imag parts of trainimage fft
        real_FFT_trainImage = FFT_trainImage.real
        imag_FFT_trainImage = FFT_trainImage.imag
        
        # real parts and imag parts of testimag fft
        real_FFT_testImage = FFT_testImage.real
        imag_FFT_testImage = FFT_testImage.imag
        
        real_FFT_trainImage = np.reshape(real_FFT_trainImage, (-1, 28, 28, 1))
        imag_FFT_trainImage = np.reshape(imag_FFT_trainImage, (-1, 28, 28, 1))
        real_FFT_testImage = np.reshape(real_FFT_testImage, (-1, 28, 28, 1))
        imag_FFT_testImage = np.reshape(imag_FFT_testImage, (-1, 28, 28, 1))
        
        return real_FFT_trainImage, imag_FFT_trainImage, real_FFT_testImage, imag_FFT_testImage
    
    
    def exploitLabel (self):
        
        output1 = self.trainLabel
        output2 = self.testLabel
        
        return output1, output2

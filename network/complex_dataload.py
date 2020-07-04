from module import *


class complex_mnist():
    
    def __init__ (self):
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
        self.x_train = self.x_train / 255.
        self.x_test  = self.x_test / 255.
        print("Get original datasets")
        
    def fourier_transform (self, inputs):
        
        outputs = np.fft.fft2(inputs)
        print("complete 2d fft")
        
        return outputs
    
    def to_categorical (self, inputs):
    
        outputs = to_categorical(inputs)
        print ("complete ont hot encoding")
        
        return outputs
    
    def normalize (self, inputs):
        
        absoulte = np.max(np.abs(inputs))
        outputs = np.divide(inputs, absoulte)
        
        return outputs
    
    def processing (self):
        
        fft_train = self.fourier_transform(self.x_train)
        fft_train = np.reshape(fft_train, [60000, 28, 28, 1])
        del (self.x_train)
        label_train = self.y_train
        del (self.y_train)
        
        fft_test = self.fourier_transform(self.x_test)
        fft_test = np.reshape(fft_test, [10000, 28, 28, 1])
        del (self.x_test)
        label_test = self.y_test
        del (self.y_test)
        
        return fft_train, label_train, fft_test, label_test


class complex_cifar10():
    
    def __init__ (self):
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
        self.x_train = self.x_train / 255.
        self.x_test  = self.x_test / 255.
        print("Get original datasets")
        
    def fourier_transform (self, inputs):
        
        outputs = np.fft.fft2(inputs)
        print("complete 2d fft")
        
        return outputs
    
    def to_categorical (self, inputs):
    
        outputs = to_categorical(inputs)
        print ("complete ont hot encoding")
        
        return outputs
    
    def normalize (self, inputs):
        
        absoulte = np.max(np.abs(inputs))
        outputs = np.divide(inputs, absoulte)
        
        return outputs
    
    def processing (self):
        
        fft_train = self.fourier_transform(self.x_train)
        del (self.x_train)
        label_train = self.to_categorical(self.y_train)
        del (self.y_train)
        
        fft_test = self.fourier_transform(self.x_test)
        del (self.x_test)
        label_test = self.to_categorical(self.y_test)
        del (self.y_test)
        
        return fft_train, label_train, fft_test, label_test
    
    
class complex_cifar100():
    
    def __init__ (self):
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar100.load_data()
        self.x_train = self.x_train / 255.
        self.x_test  = self.x_test / 255.
        print("Get original datasets")
        
    def fourier_transform (self, inputs):
        
        outputs = np.fft.fft2(inputs)
        print("complete 2d fft")
        
        return outputs
    
    def to_categorical (self, inputs):
    
        outputs = to_categorical(inputs)
        print ("complete ont hot encoding")
        
        return outputs
    
    def normalize (self, inputs):
        
        absoulte = np.max(np.abs(inputs))
        outputs = np.divide(inputs, absoulte)
        
        return outputs
    
    def processing (self):
        
        fft_train = self.fourier_transform(self.x_train)
        del (self.x_train)
        label_train = self.to_categorical(self.y_train)
        del (self.y_train)
        
        fft_test = self.fourier_transform(self.x_test)
        del (self.x_test)
        label_test = self.to_categorical(self.y_test)
        del (self.y_test)
        
        return fft_train, self.y_train, fft_test, self.y_test

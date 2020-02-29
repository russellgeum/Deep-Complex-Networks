from module import *
from generator import *
from activation import *

class complexMODEL():
    
    def __init__ (self, conv_filter1 = 16, 
                        conv_filter2 = 32, 
                        conv_filter3 = 64,
                        dense_connected1 = 4*64,
                        dense_output = 10):
        
        # Convolution 필터 갯수를 정의
        self.conv_filter1 = conv_filter1
        self.conv_filter2 = conv_filter2
        self.conv_filter3 = conv_filter3

        # Dense 레이어에 연결하는 노드 수를 정의
        self.dense_connected1 = dense_connected1
        
        # dataGenerator에서 데이터를 받고, FFT 변환된 이미지와 라벨 따오기
        self.data = dataGenerator()
        self.trainReal, self.trainImag, self.testReal, self.testImag = self.data.FFTprocessing()
        self.trainLabel, self.testLabel = self.data.exploitLabel()
        
        # multi input으로 넣기 위해 복소수 이미지들을 리스트로 묶어줌
        self.trainData = [self.trainReal, self.trainImag]
        self.testData = [self.testReal, self.testImag]
        
        
        # 필요한만큼 convolution 레이어 인스턴스를 생성
        self.Conv1 = complexConv2D (self.conv_filter1)
        self.Conv2 = complexConv2D (self.conv_filter2)
        self.Conv3 = complexConv2D (self.conv_filter3)
        
        # 필요한 만큼 dnese 레이어 인스턴스를 생성
        self.Dense1 = complexDense(self.dense_connected1)
        self.Dense2 = complexDense(dense_output)
        
        
        
    """
    train 메서드에 들어가는 arguments들
    fit(x=None, 
    y=None, 
    batch_size=None, 
    epochs=1, 
    verbose=1, 
    callbacks=None,
    validation_split=0.0, 
    validation_data=None, 
    shuffle=True, 
    class_weight=None,
    sample_weight=None, 
    initial_epoch=0, 
    steps_per_epoch=None,
    validation_steps=None, 
    validation_freq=1,
    max_queue_size=10, 
    workers=1,
    use_multiprocessing=False)
    """
    def train(self, batch_size = 512, 
                    epochs = 50, 
                    verbose = 1, 
                    callbacks = None,
                    validation_split = 0.0, 
                    validation_data = None, 
                    shuffle = True, 
                    class_weight = None,
                    sample_weight = None, 
                    initial_epoch = 0, 
                    steps_per_epoch = None,
                    validation_steps = None, 
                    validation_freq = 1,
                    max_queue_size = 10, 
                    workers = 1,
                    use_multiprocessing = False):
        
        input1 = Input(shape = (28, 28, 1), name = "real")
        input2 = Input(shape = (28, 28, 1), name = "imag")

        real, imag = self.Conv1.fowardConvolution(input1, input2)
        real, imag = CReLU(real, imag)
        real, imag = self.Conv2.fowardConvolution(real, imag)
        real, imag = CReLU(real, imag)
        real, imag = self.Conv3.fowardConvolution(real, imag)
        real, imag = CReLU(real, imag)
        real, imag = Flatten(real, imag)
        real, imag = self.Dense1.fowardDense(real, imag)
        real, imag = CReLU(real, imag)
        real, imag = self.Dense2.fowardDense(real, imag)
        real, imag = Softmax(real, imag)

        outputs = real + imag

        model = Model(inputs = [input1, input2],
                    outputs = outputs, 
                    name='mnist_model')
        
        
        """
        one hot vector로 이미 만들어져 있으면 categorical crossentorpy
        그렇지 않으면 sparse categorical crossentropy
        """
        model.compile(optimizer = Adam(learning_rate = 0.001),
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ["accuracy"])
        model.summary()
        model.fit(self.trainData, 
                    self.trainLabel,
                    batch_size = batch_size, 
                    epochs = epochs, 
                    verbose = verbose,
                    validation_data = (self.testData, self.testLabel))
        
        return model


if __name__ == '__main__':
    test = complexMODEL()
    test.train(verbose = 1)
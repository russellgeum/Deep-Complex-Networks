# Deep Complex Networks (아래 두 개는 참고)  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
#
# About Deep Complex Networks
1. Models of ordinary deep learning update parameters in the real numbers area.
2. But should it be real number? Can you think about deep learning model in the field of complex numbers?
3. Deep learning in complex numbers has more expressive power than in the real numbers. (It is said that there are many advantages.)
4. This paper introduces the neural network module in the field of multiple numbers, and introduces some active functions as possible.
In this repo, the solution networks and proposed activation functions are implemented.  
It then examines the performance of active functions in the complex numbers and examines the following mathematical issues.  
#
# Datasets  
개념에 적용하기 쉬운 MNIST로 먼저 테스트, 이후 Cifar-10, Cifar-100으로 테스트  
이미지를 discrete fourier transform을 하여 복소수 도메인으로 변환  
#
# 네트워크의 unit  
![unit](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/unit.png)  
## Convolution Networks  
![cnn](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/architect_picture.png)  
## Activation Functions  
![act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation_concept.png)  
#
#
복소수들이 이 활성 함수에 입력되면 어떤 모양이 되는지, 직접 활성 함수를 구현하여 출럭값의 분포를 그려보자  
800여 개의 복소수들을 준비하고, 각각의 활성 함수에 입력하였다. 순서대로 CReLU, zReLU, modReLU이다.  
![Act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation_result.png)
### Activation function 특징  
- activation function은 preserve region에서 Cauchy–Riemann equations 조건을 만족한다.  
- CReLU에서는 2사분면 복소수는 허수부 값만 취하고, 4사분면 복소수는 실수부 값만 취한다.  
- zReLU에서는 2사분면, 4사분면 복소수 모두 버린다.  
- 3사분면 복소수는 두 활성 함수에서 모두 버린다.  
#
## 이미지에 2D FFT를 적용한 모습
왼쪽은 실수부, 오른쪽은 허수부
![sample](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/2D%20FFT.png)  
#
## Activation function 테스트
### 테스트 환경
- Adam(learning_rate = 0.001) && RMSprop(learning_rate = 0.001)  
- Batch size = 512, Epoch = 20, same architecture  
### adam optimizer에서 테스트
![a1](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20CReLU.png)  
![a2](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20zReLU.png)  
![a3](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20modReLU.png)  
### RMSprop optimzier에서 테스트
사진에서 오류 정정 : 순서대로 CReLU, zReLU, modReLU 사진임  
![R1](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20CReLU.png)  
![R2](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20zReLU.png)  
![R3](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20modReLU.png)  
#
- 셋 중 zReLU가 가장 안정적인 성능을 보임  
- RMSprop에서의 modReLU는 Adam에 비해 loss 변동이 큼  
- modReLU는 성능이 그리 좋지 못한 것 같다. (일단은 논문의 견해와 일치)  
# 차후에 수행할 실험과 생각할 것
- Cifar10, Cifar100을 이용하여 더 큰 신경망에서 논문대로의 성능이 나오는지 확인할 필요  
- LSTM 모델에서는 Cauchy–Riemann equations을 만족하는 activation function의 존재성에 대한 질문

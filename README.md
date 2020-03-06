# 논문, (2번, 3번은 추가 참고)  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
#
# Deep Complex Networks의 요약  
1. 보통의 딥러닝의 모델들은 실수 영역에서의 파라미터를 업데이트 한다.  
2. 그런데 꼭 실수여야 할까? 복소수 영역에서의 딥러닝 모델을 생각할 수 있을까?  
3. 복소수 영역에서의 딥 러닝이 실수 영역보다 풍부한 표현력을 가지고 있다. (장점이 많다고 한다.)  
4. 이 논문에서는 복소수 영역에서의 신경망 모델을 소개하고, 가능한 몇 가지 활성 함수를 소개.  
이 repo에서는 그 중 convolution networks와 제안된 activation functions들을 구현.  
그리고 복소수 영역에서의 활성 함수 성능을 살펴보고 따르는 수학적 이슈를 검토.  
#
# 사용한 데이터셋  
개념에 적용하기 쉬운 MNIST로 먼저 테스트  
이후 Cifar-10, Cifar-100으로 테스트  
이미지를 discrete fourier transform을 하여 복소수 도메인으로 변환  
#
# Complex Networks의 구조 (Convolution 위주로)  
## Convolution Networks, Residual Network  
![cnn](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/architect_picture.png)  
## 제안된 Activation Functions  
![act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation_concept.png)  
#
복소수들이 이 활성 함수에 입력되면 어떤 모양이 되는지, 직접 활성 함수를 구현하여 출럭값의 분포를 그려보자  
800여 개의 복소수들을 준비하고, 각각의 활성 함수에 입력하였다. 순서대로 CReLU, zReLU, modReLU이다.  
![Act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation_result.png)
### 활성 함수들의 특징  
- activation function은 preserve region에서 Cauchy–Riemann equations 조건을 만족한다.  
- CReLU에서는 2사분면 복소수는 허수부 값만 취하고, 4사분면 복소수는 실수부 값만 취한다.  
- zReLU에서는 2사분면, 4사분면 복소수 모두 버린다.  
- 3사분면 복소수는 두 활성 함수에서 모두 버린다.  
#
## 2D FFT를 적용한 모습  
![sample](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/2D%20FFT.png)  
#
## 활성 함수 별로 테스트
### 테스트 환경
- Adam(learning_rate = 0.001) && RMSprop(learning_rate = 0.001)  
- Batch size = 512, Epoch = 20, same architecture  
### adam optimizer에서 테스트
![a1](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20CReLU.png)  
![a2](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20zReLU.png)  
![a3](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/adam%20modReLU.png)  
### RMSprop optimzier에서 테스트
![R1](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20CReLU.png)  
![R2](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20zReLU.png)  
![R3](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/rmsp%20modReLU.png)  
#
- 셋 중 zReLU가 가장 안정적인 성능을 보임  
- modReLU는 이 조건에서는 가장 성능이 낮음  
- RMSprop에서의 modReLU는 Adam에 비해 loss 변동이 큼  
- modReLU는 성능이 그리 좋지 못한 것 같다. (일단은 논문의 견해와 일치)
# 차후에 수행할 실험과 생각할 것
- Cifar10, Cifar100을 이용하여 더 큰 신경망에서 논문대로의 성능이 나오는지 확인할 필요  
- LSTM 모델에서는 Cauchy–Riemann equations을 만족하는 activation function의 존재성에 대한 질문
- Batch normalization과 Pooling 등 기타 모듈은 차후에 구현

# 논문, 아래 두 개는 추가 참고  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
#
# Deep Complex Networks의 요약  
1. 보통의 딥러닝의 모델들은 실수 영역에서의 파라미터 업데이트 연산을 한다.  
2. 그런데 꼭 실수여야 할까? 복소수 영역에서의 딥러닝 모델을 생각할 수 있을까?  
3. 복소수 영역에서의 딥 러닝이 실수 영역보다 풍부한 표현력을 가지고 있다. (장점이 많다고 한다.)  
4. 이 논문에서는 복소수 영역에서의 신경망 모델을 소개하고, 가능한 몇 가지 활성 함수를 소개한다.  
나는 그 중 convolution networks와 제안된 activation functions들을 구현한다.  
그리고 복소수 영역에서의 활성 함수가 어떠한 수학적 이슈가 따르는지 검토할 것이다.  
#
# 사용한 데이터셋  
우리에게는 개념에 적용하기 쉽고, 다루기 편리한 데이터셋인 MNIST가 있다.  
numpy.fft.fft2 메서드로 discrete fourier transform을 하여 복소수 데이터를 획득  
이를 이용하여 복소수 영역에서의 이미지 분류 문제를 만들어보자.  
#
# Complex Networks의 구조 (Convolution 위주로)  
## Convolution Networks, Residual Network  
![cnn](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/architect1.PNG)  
## 제안된 Activation Functions  
![act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation_concept.png)  
#
복소수들이 이 활성 함수에 입력되면 어떤 모양이 되는지, 직접 활성 함수를 구현하여 값의 분포를 그려보자  
800여 개의 복소수들을 준비하고, 각각의 활성 함수에 입력하였다. 순서대로 CReLU, zReLU, modReLU이다.  
![Act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation2.png)
#
활성 함수들의 특징    
- CReLU에서는 2사분면 복소수는 허수부 값만 취하고, 4사분면 복소수는 실수부 값만 취한다.  
- zReLU에서는 2사분면, 4사분면 복소수 모두 버린다.  
- 3사분면 복소수는 두 활성 함수에서 모두 버린다.  
- 활성 함수는 preserve region에서 Cauchy–Riemann equations 조건을 만족한다.  
#
## 2D FFT를 적용한 모습  
![sample](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/2D%20FFT.png)  

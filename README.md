# Reference  
- Deep Complex Networks (ICLR 2018)  
  URL : https://arxiv.org/abs/1705.09792  
- On Complex Valued Convolutional Neural Networks  
  URL : https://arxiv.org/abs/1602.09046  
- Complex domain backpropagation  
  IEEE transactions on Circuits and systems II: analog and digital signal processing, 39(5):330–334, 1992.  
  
# 요약  
1. 이제까지 딥러닝의 모델들은 실수 영역에서의 파라미터 업데이트를 목표로 한다.  
2. 그런데 논문에서는 복소수 영역에서의 딥러닝 모델의 잠재적인 가능성을 이야기한다.  
3. 복소수 영역에서의 딥 러닝이 실수 영역보다 풍부한 표현력을 가지고 있음을 이전의 논문들에서 알 수 있다.  
4. 이러한 생각을 차용한 예시는 신호를 푸리에 변환을 하여 합성곱 신경망을 수행한 것이 있기도 하다.  
5. 이 논문에서는 복소수 영역에서의 신경망 모델을 소개하고, 가능한 몇 가지 활성 함수를 소개한다.  
나는 그 중 Convolution Networks를 구현한다. 
  
# 데이터셋  
우리에게는 개념에 적용하기 쉽고, 다루기 편리한 데이터셋인 MNIST가 있다.  
이를 이용하여 복소수 영역에서의 이미지 분류 문제를 만들어보자.  

# Complex Networks의 구조 (Convolution 위주로)  
## Convolution Networks, Residual Networks  
![cnn](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/architect1.PNG)  
## Activation Functions  
![act](https://github.com/Doyosae/Deep_Complex_Networks/blob/master/images/activation1.PNG)  

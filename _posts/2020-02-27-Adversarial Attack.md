---
title: Adversarial Attack 동향 및 이를 활용한 Image Recognition 증가
description: Adversarial Attack Image
categories:
 - paper
tags:
 - Adversarial Attack, AdvProp , Adversarial Examples Improve Image Recognition
---
Image 학습 분야에서 Deep Learning 을 활용한 다양한 NN 기술들이 등장하면서 다양한 SOTA 가 나오고 있지만 
이와 같은 분야에서 Adversarial Attack 에 취약하다는 사실이 알려지고 이를 이용하여 Training 을 해보자.  
## Adversarial Attack 이 뭐야?
원본 이미지에 임의의 값을 추가해도 실제 원본 이미지 이지만 분류상 다르게 찾는 경우
![](https://openai.com/content/images/2017/02/adversarial_img_1.png)
57.7% 확률로 판다라고 판단했던 이미지에 noise 를 추가하는 99.3% 확률로 gibbon 이라 찾아내는 이런 어리석음이란...

## Adversarial Image 는 어떻게 생성해?
#### Fast Gradient Signed Method (FGSM) 기반 
수식만 살펴보면 다음과 같다.
![](https://miro.medium.com/max/685/1*p0CybspN89jSzmM4V4Ovng.png)
X = Image

Y = GroundTruth 

∇x = X gradient

J = Cost Function

θ = Network Parameter

## Tensorflow Example (sudocode)
~~~python
loss = model.network()
self.grad = tf.gradient(loss,model.x_input)[0]

grad = sess.run(self.grad , feed_dict={self.model.x_input : x , self.model.y_input : y})
x += self.eps * numpy.sign(grad)
# x == new Image
~~~ 

## Adversarial 을 이용한 network improve 방법
[Adversarial Examples Improve Image Recognition](https://arxiv.org/abs/1911.09665)

[관련 post]()
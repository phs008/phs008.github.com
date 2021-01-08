---
title: Searching for Activation Functions
description: Activation Functions (aka) Swish
categories:
 - paper
tags:
 - activation function
toc: true
---
## Searching for Activation Function
* 출처 : https://arxiv.org/abs/1710.05941
* date : Fri, 27, Oct 2017
### Intro
* RL 기반 Activation Function 을 찾아보자!
* ReLU를 대체할만한 Swish 라는 Activation Function 을 발견
* Search space 를 활용하여 찾음
### Result
* 자세한 성능 관련 이야기는 생략한다. (어짜피 좋다고 말하는게 뻔하니깐)
* 기존 ReLU 사용할때 보다 lr 을 낮추는게 잘동작한다 말함.
### Swish
```
x = x * sigmoid(beta * x)
or 
tf.nn.swish(x)
```
### Swish 와 Sigmoid 차이
![](https://user-images.githubusercontent.com/17635409/82277484-eb621400-99c2-11ea-9640-0e700892bf3b.png)
GeoGebra : https://www.geogebra.org/m/rsscdr7j
### 장점
* 기존 Activation 을 대체하기가 아주아주 쉽다.





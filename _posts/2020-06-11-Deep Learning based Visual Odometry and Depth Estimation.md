---
title: 딥러닝을 활용한 Visual Odometry 와 Depth Estiamtion 
description: 
categories:
 - visual odometery , depth estimation
tags:
 - visual odometery , depth estimation
 toc: true
---
# Depth estimation and Visual Odometry

## 들어가기에 앞서
본 자료는 다음의 사이트의 자료를 기반으로 구성함

[[reference]](https://goodgodgd.github.io/ian-flow/)


## 용어설명

+ visual odometry
    
    카메라 이미지를 분석하여 위치와 방향을 결정 하는 프로세스  [[reference]](https://g.co/kgs/X8tqQM)
+ Depth Estimation
    
    영상 으로부터 영상의 깊이를 예측하는 기술.
    
    고전적 방식으로 스트레오 이미지를 통한 SFM (Structure from motion) 이 존재함. [[reference]](https://g.co/kgs/DVnDkq)
    
    또는 ToF 센서등을 통해 측정 가능
    
## 기술동향

+ vSlam (Visual Simultaneous localization and Mapping)
 
    과거엔 로봇 바퀴에 달린 인코더나 IMU 를 이용하여 자세변화를 계산하였고 이와 유사한 연구분야로 vSLAM 이라는 기술이 있는데 이는 자세의 변화량을 누적시키는 Visual Odometry 에 누적위치 오차를 loop closing 으로 해소하는 형태를 보여준다.
    ![](https://www.researchgate.net/profile/Sherine_Rady2/publication/311948486/figure/fig9/AS:668983304925186@1536509457066/Loop-closing-example-Ho-and-Newman-2007-a-A-snapshot-of-a-SLAM-just-before-loop.png)
    
    + 고전적(?) 방식으로는 이미지로부터 Feature extraction , descriptor 을 이용하여 두 영상 사이 특징점을 매칭하고 이를 통해 영상 사이의 자세변화 를 알아내는 방법이다.
        + orb-slam 등이 대표적     

+ Depth Estimation
    
    한장의 이미지로부터 각 픽셀별 깊이를 추정하는 기술. 
    
## Deep Learning approach in Depth Estimation

Depth 를 추정하는 방식을 deeplearning 기반으로 해보자.
왜냐면 cnn 기반 deeplearning approach 는 결국 feature extractor (filters) 를 학습 시키는거니깐.
+ supervised learning method
    
    가장 대표적인 논문은 'Depth Map Prediction from Single Image using a Multi-Scale Deep Network' (aka. Eigne)(NIPS, 2014)
    
    RGB-D 카메라를 이용하여 스트레오 이미지와 depth gt 를 통해 DNN 모델을 학습하였다.
    
    하지만 supervised learning 상 gt 를 매번 얻어야 함에 따라 unsupervised learning approach 가 점점 많아지고 있다.
    
+ unsupervised learning method
    
    더이상 gt 는 필요없다! 'Unsupervised monocular depth estiamtion with left-right consistency' (CVPR, 2017) 논문이 나왔는데 이를 줄여서 monoDepth 라고 부른다.
    
    위 논문은 스트레오 영상을 이용하여 학습을 진행하는것인데 원리는 다음과 같다.
    
    1. 스트레오 카메라에서 두 카메라 사이 거리를 알고
    2. 왼쪽 영상의 depth ( 또는 **disparity** ) 를 안다면
    3. 오른쪽 시점의 이미지를 왼쪽 이미지로 부터 재구성 할수 있다.
        
    쉽게 말해 왼쪽 이미지를 오른쪽 이미지 시점으로 재구성하여 이미지가 같아지도록 학습하고 오른쪽 이미지는 왼쪽 이미지 시점으로 재구성하여 이미지가 같아지도록 학습하는 방식.
    
    이때 재구성한 이미지와 원래 이미지 사이의 차이를 **photometric loss* 라고 말하는데 앞으로 많은 논문에 다음의 용어가 쓰이고 주요 loss 가 된다.
    

## Depth 추정을 위한 수학적 접근방법 

[computer vision basic](https://phs008.github.io/visual%20odometery%20,%20depth%20estimation/2020/06/11/Computer-Vision-Basic-for-Depth-estimation/)



## etc..

추후 블로깅 예정 논문 : [sfmlearner](Unsupervised Learning of Depth and Ego-Motion from Video)

 
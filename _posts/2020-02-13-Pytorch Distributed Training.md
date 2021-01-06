---
title: Pytorch Distributed Training 방법
description: PyTorch 를 이용한 분산환경 Training 방법
categories:
 - PyTorch
tags:
 - PyTorch
 - Distributed Training
---
## PyTorch Distributed Training 
### Distributed Training
N 개 이상의 GPU 서버를 활용하여 동시에 학습을 진행함을 의미한다.
### Communication backend
#### TCP
일반적으로 알고 있는 TCP Protocol 을 활용하여 서버간 Communication 을 수행한다.
다만 GPU에 대한 지원을 할수 없다. 
#### Gloo
[**GLOO**](https://github.com/facebookincubator/gloo) 는 Facebook 에서 개발한 collective commnuication library 이며 CPU 와 GPU를 동시 사용한다.
또한 [**NCCL**](https://developer.nvidia.com/nccl) 를 사용하여 멀티 GPU 노드간 통신을 수행할수 있고 이런 노드간 루틴을 위한 자체 알고리즘을 포함하고 있다.
#### MPI
[**MPI**]()(Message Passing Interface) 는 이미 예전부터 고성능 컴퓨팅 분야 표준도구로 사용되던 백엔드 이다.
다만 PyTorch 는 MPI 구현을 포함하지 않고 수동으로 컴파일을 해야 한다.
* PyTorch 소스로 설치하고 MPI 백인대 설치 하는 방법.
```
1. 아나콘다 환경을 만들고 활성화하고, ` 가이드 <https://github.com/pytorch/pytorch#from-source>`__ 에 따라 모든 필수 조건을 설치한다. 그러나 아직 python setup.py install 을 실행하진 않는다
2. 원하는 MPI 구현을 선택하고 설치하십시오. CUDA 인식하는 MPI를 활성화하려면 몇 가지 추가 단계가 필요하다 . 다음방법은  GPU 없이 Open-MPI를 사용 하는 예제이다: conda install -c conda-forge openmpi
3. 이제 복제 된 PyTorch repo 로 이동하여 python setup.py install 을 실행한다.
```

#### reference
[Pytorch 로 분산 어플리케이션 개발](https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/intermediate/dist_tuto.html#setup) 
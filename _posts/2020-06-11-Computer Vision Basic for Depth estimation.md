---
title: 깊이 추정을 위한 컴퓨터 비전 기본 수학
description: 
categories:
 - visual odometery , depth estimation
tags:
 - visual odometery , depth estimation
toc: true
---
# Computer vision basic for depth estimation

## 들어가기에 앞서
본 자료는 다음의 사이트의 블로그 를 그대로 첨부함.
https://goodgodgd.github.io/ian-flow/


## Computer Vision Basic

사람의 눈이 빛을 받아들여 시신경에서 빛의 세기를 감지하여 뇌에서 이를 종합하여 영상을 만들어 내듯이 카메라도 렌즈를 통해 들어온 빛을 작은 기판위에 밀집된 미세한 센서들이 감지하여 영상을 만들어냅니다. 아래 그림은 전형적인 핀홀(pinhole) 카메라 모델의 개념도입니다.  

![pinhole-model](http://phs008.github.io/assets/2020-06-11/pinhole-model.png)

그림에서 카메라는 Z 축을 향하고 있고 그 앞에 ```mathbf{X}``` 라는 점이 있습니다. 원점과 점 사이에는 `image plane`이 있는데 이것은 실제 센서라기 보다는 3차원 형상이 2차원 영상으로 투영되는 가상의 평면입니다. 3차원 상의 점들은 각각의 색(color)을 가지고 있고 그 점이 원점을 향하면서 만나는 `image plane`의 좌표($$x,y$$)에 그 색이 맺혀 영상이 만들어지는 것입니다. 영상을 나타내는 점의 단위를 픽셀(pixel)이라 하는데 이미지 센서에는 픽셀마다 Red, Blue, Green 세 가지 색의 세기(intensity)를 측정할 수 있는 센서가 세 개씩 배치되어 있고 그 세개의 영상을 모으면 우리가 보는 RGB 영상, 즉 컬러 영상이 되는 것입니다.  

![rgb-channels](http://phs008.github.io/assets/2020-06-11/rgb-channels.jpg)

이때 물체의 3차원 좌표와 이 점이 영상에 맺히는 픽셀 좌표 사이에는 밀접한 관계가 있습니다. 3차원 좌표$$(X,Y,Z)$$와 픽셀 좌표($$x_{img}, y_{img}$$) 사이의 관계는 핀홀 카메라 모델 그림의 왼쪽을 보면 다음과 같이 비례식으로 쉽게 구할 수 있습니다.  


```
x_{img} = f_x \left( X \over Z \right) + c_x  
y_{img} = f_y \left( Y \over Z \right) + c_y
```


위 식에서 $$f_x, f_y$$는 원점과 `image plane` 사이의 거리를 나타내는 focal length (초점 거리)고, 사각형의 `image plane`에서는 왼쪽 위가 원점인데 Z축이 통과하는 영상의 중심점의 좌표가 $$c_x, c_y$$ 입니다. 이를 벡터식으로 표현하면 카메라 파라미터들을 하나의 행렬에 담을 수 있는데 이 행렬 $$K$$를 camera projection matrix라 합니다. 간단히 말하면 3차원 좌표에 camera projection matrix를 곱하면 픽셀 좌표가 나온다고 볼 수 있습니다.


$$
\begin{bmatrix} x_{img} \\ y_{img} \\ 1 \end{bmatrix} =
{1 \over Z } \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} 
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} \\
\mathbf{p} = {1 \over Z } K \mathbf{P}
$$


이처럼 3차원 좌표와 영상의 픽셀 좌표 사이에는 비례/반비례 관계가 있기 때문에 두 영상에서 똑같은 점들이 다른 픽셀에 보일 때 픽셀의 이동량을 통해 카메라의 움직임을 추정할 수 있습니다. 이것이 Visual Odometry의 기본 원리인데 여기서는 딥 러닝 기반의 접근 방식을 다루기 때문에 기존 VO의 세부적인 내용은 넘어가도록 하겠습니다.



## 5. Pose Representation

VO를 학습을 통해 해결할 수 있다고 했는데 VO-DNN의 출력은 어떻게 나와야 할까요? 두 프레임 사이의 상대적인 자세를 어떻게 표현해야 할까요?  

![two-view-geometry](http://phs008.github.io/assets/2020-06-11/two-view-geometry.jpg)

두 개의 카메라가 (혹은 한 카메라가 이동한 두 개의 시점이) 있습니다. 이들 사이의 상대적인 자세(pose)는 두 카메라 사이의 상대적인 이동(translation)과 회전(rotation)으로 표현할 수 있습니다. 이를 다르게 표현하면 두 카메라 사이의 상대 pose는 두 카메라 좌표계 사이의 **Rigid Transformation** (translation + rotation)으로 표현할 수 있다는 뜻입니다.  

3차원 컴퓨터 비전에서 변환(Transformation)이란 주로 좌표계 변환을 의미합니다. 3차원 공간 상의 점(point)은 그 위치 자체를 의미하지만 점의 위치를 정확히 표현하기 위해서는 $$(X, Y, Z)$$와 같은 좌표(coordinates)를 써야 합니다. 즉 특정 기준 좌표계로부터의 상대적인 위치를 숫자로 표현해야 합니다. 위 그림에서 점은 하나지만 좌표계는 두 개가 있습니다. 똑같은 점이 왼쪽 카메라 좌표계와 오른쪽 카메라 좌표계에서는 다른 좌표로 표현이 됩니다. 한쪽 좌표계에서 본 점의 좌표를 다른 쪽 좌표계로 변환하는 것이 바로 좌표계 변환입니다. 두 좌표계 사이의 좌표 변환은 이동과 회전을 통해 계산할 수 있고 각 카메라 마다 좌표계를 가지고 있으므로 카메라의 자세를 좌표계 변환으로 표현할 수 있습니다.

이동은 직관적으로 두 카메라 사이의 상대 위치를 3차원 벡터로 표현하면 되지만 회전을 표현하는 방법은 제가 알기로 최소 네 종류가 있습니다. 이에 따라 rigid transformation도 네 가지로 표현할 수 있습니다.  



### 5.1 Special Eucliean Transformation

카메라의 pose는 rigid transformation (or rigid body motion)으로 표현할 수 있다고 했는데 rigid transformation을 좀더 기술적인(?) 용어로 말하면 **Special Euclidean Transformation** 입니다.  

일단 앞에 "special"은 떼고 Euclidean Transformation만 정의해보면 다음과 같습니다. 쉽게 말해 모양이 늘어나거나 줄어들거나 뒤틀리지 않고 똑같이 유지된다는 것이죠.

> **Euclidean transformation**: a map that preserves the Euclidean distance between every pair of points. The set of all Euclidean transformation in 3-D space is denoted by $ E(3) $.
> $$
> g: \mathbb{R}^3 \to \mathbb{R}^3; X \mapsto g(X) \\
> \begin{Vmatrix} g_*(v) \end{Vmatrix} = \begin{Vmatrix} v \end{Vmatrix}, \forall v \in \mathbb{R}^3
> $$
>

"Special" Euclidean Transformation은 여기에 한가지 조건을 더 붙입니다. 방향이 바뀌지 않아야 한다는 것이죠.  

> The map or transformation induced by a rigid-body motion is called a **special Euclidean transformation**. The word "*special*" indicates the fact that a transformation is **orientation-preserving**. 
>

$$
g_*(u) \times g_*(v) = g_*(u \times v), \forall u, v \in \mathbb{R}^3
$$

아무리 모양을 유지시킨채 회전을 시켜도 좌우반전은 일어나지 않습니다. 상하반전도 마찬가지입니다. 좌우반전이나 상하반전 같은 변환은 Euclidean Transformation은 만족하지만 물리적인 3차원 공간에서는 일어날 수 없는 일입니다. 우리가 일반적으로 쓰는 직교 좌표계는 $$\mathbf{X} \times \mathbf{Y} = \mathbf{Z}$$ 를 만족하는데 이 좌표계를 아무리 "회전"시켜도 저 등식은 성립합니다. 하지만 좌우반전으로 X축만 뒤집으면 ($$\mathbf{X'} = -\mathbf{X}$$) 변환된 좌표계에서는 $$\mathbf{X'} \times \mathbf{Y} = -\mathbf{X} \times \mathbf{Y} = -\mathbf{Z} $$ 이므로 이를 만족할 수 없게됩니다.  

그래서 Euclidean Transformation 중에 **orientation-preseving**한 부분 집합이 Special Euclidean Transformation이고 이것이 물리 세계의 강체 변환(Rigid Transformation)과 동일한 의미를 갖게 됩니다. (강체(rigid body)란 단단해서 모양을 변형할 수 없는 물체이므로 당연히 좌우반전도 할 수 없습니다.)  

이제 3-D Special Euclidean Transformation, 줄여서 **"SE(3)"**라 표기하는 transformation 혹은 pose를 표현하는 방법은 여러가지가 있습니다. 이동(translation)은 공통적으로 3차원 벡터로 표현하지만 회전(rotation)을 표현하는 방법에 따라 달라집니다. 다음은 회전을 표현하는 네 가지 방법입니다.

- Rotation matrix
- Euler angle
- Quaternion
- Twist coordinates

이제 저 표현들을 하나씩 알아봅시다.



### 5.2 Rotation matrix

회전(rotation) 변환은 아래 그림처럼 원점은 그대로 둔채 특정 축을 중심으로 좌표계를 회전시키는 변환입니다. 

![rotation_rigid_body](http://phs008.github.io/assets/2020-06-11/rotation_rigid_body.png)



회전 변환은 3x3 matrix로 표현할 수 있는데 임의의 3x3 matrix가 모두 **rotation matrix** 가 될 수 있는건 아닙니다. 3x3 matrix 중에서 rotation matrix가 될 수 있는 matrix 집합을 **Special Orthogonal group, SO(3)**라고 부르고 집합의 조건은 다음 식의 (1)과 같습니다. (2), (3)은 special euclidean trnasformation의 조건을 회전변환으로 다시 쓴 것입니다.


$$
(1)\quad SO(3) \doteq \left\{ R \in \mathbb{R}^{3\times3} \middle|\ R^TR=I,\ det(R)=+1 \right\} \\
(2)\quad \left| \mathbf{x} \right| = \left| R\mathbf{x} \right| \\
(3)\quad (R\mathbf{x_1}) \times (R\mathbf{x_2}) = R(\mathbf{x_1} \times \mathbf{x_2})
$$


$$R^TR=I$$ 조건을 만족하는 matrix 집합을 orthogonal group, $$O(3)$$라고 하는데 이는 euclidean transformation의 회전에 해당합니다. Orthogonal matrix는 (2)처럼 변환이 벡터의 크기에 영향을 주지 않습니다.   

Orthogonal group에 $$det(R)=+1$$ 조건까지 더해야 비로서 (3)을 만족할 수 있고 이것이 special euclidean transformation에 해당하는 "special" orthogonal group (a.k.a rotation matrix) 입니다.  

Rotation matrix ($$R$$)에 translation vector ($$\mathbf{t}$$)를 더해 완성한 SE(3)의 변환 행렬(transformation matrix)는 다음과 같습니다.


$$
SE(3) \doteq \left\{ 
T = \begin{bmatrix} R & \mathbf{t} \\ 0 & 1 \end{bmatrix}
\middle\|\ R \in SO(3), \mathbf{t} \in \mathbb{R}^3 
\right\} 
\in \mathbb{R}^{4 \times 4}
$$



### 5.3 Euler angle

Euler angle이란 임의의 회전을 세 번의 직교 좌표계 축 회전으로 표현하는 방법입니다. 가장 간단예로 xyz euler angle이 있습니다. 직교 좌표계에서 x축으로 $$\alpha$$, y축으로 $$\beta$$, z축으로 $$\gamma$$ 만큼 회전하면 어떤 회전이라도 표현할 수 있다는 것입니다.


$$
\forall R \in SO(3), \exists\ \alpha, \beta, \gamma \\
R=R_x(\alpha)R_y(\beta)R_z(\gamma)
$$

$$
R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 
0 & cos\alpha & sin\alpha \\
0 & -sin\alpha & cos\alpha \end{bmatrix} \quad 
R_y(\beta) = \begin{bmatrix} cos\beta & 0 & -sin\beta \\ 
0 & 1 & 0 \\
sin\alpha & 0 & cos\alpha \end{bmatrix} \quad 
R_z(\gamma) = \begin{bmatrix} cos\gamma & sin\gamma & 0 \\ 
-sin\gamma & cos\gamma & 0 \\
0 & 0 & 1 \end{bmatrix}
$$

위의 rotation matrix의 의미는 다음과 같습니다.

1. Coordinate system rotation: 점은 가만히 있고 좌표축이 회전한 경우 $$R\mathbf{p}$$ 연산을 통해 회전된 좌표계에서의 좌표를 구할 수 있습니다. 전역 좌표계에서 지역 좌표계로 들어가는 변환으로 볼 수 있습니다. 


위키를 찾아보면 [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix)에 나오는 행렬들과는 부호가 다릅니다. 위키에 나오는 회전 행렬은 다음과 같습니다.

$$
R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 
0 & cos\alpha & -sin\alpha \\
0 & sin\alpha & cos\alpha \end{bmatrix} \quad 
R_y(\beta) = \begin{bmatrix} cos\beta & 0 & sin\beta \\ 
0 & 1 & 0 \\
-sin\alpha & 0 & cos\alpha \end{bmatrix} \quad 
R_z(\gamma) = \begin{bmatrix} cos\gamma & -sin\gamma & 0 \\ 
sin\gamma & cos\gamma & 0 \\
0 & 0 & 1 \end{bmatrix}
$$

위키의 rotation matrix의 의미는 두 가지로 볼 수 있습니다.

1. Point rotation: 좌표계는 가만히 있고 점이 원점을 중심으로 회전한 경우 $$R\mathbf{p}$$ 연산을 통해 회전된 좌표를 구할 수 있습니다. 
2. **Pose representation:** transformation matrix로 pose를 표현할 때 $$R\mathbf{p}$$ 연산을 통해 해당 pose에서 본 지역 좌표를 전역 좌표로 변환할 수 있습니다. 사실 이것은 앞서 나온 좌표계 회전으로 해야하는데 카메라가 $$\theta$$ 만큼 회전되어 있을 때 이를 전역 좌표로 변환하려면 좌표계 회전을 $$-\theta$$ 만큼 해야해서 결과적으로 점 회전(point rotation)과 같아집니다.



Euler angle 사용시 회전 순서를 꼭 xyz 순서로 할 필요는 없고 아래 그림과 같은 zxz euler angle도 많이 쓰입니다. 그외에도 여러가지 조합이 있는데 연속으로 같은 축에 대해서 회전하지만 않으면 어떠한 조합도 euler angle이 될 수 있습니다. 예를 들어 xxy, zyy는 안되지만 xyx, yxz는 됩니다.

| ![euler-zxz-rotation](http://phs008.github.io/assets/2020-06-11/euler-zxz-rotation.gif) |
| ----- |
| zyz euler angle rotation, [출처](https://hoodymong.tistory.com/3) |



Euler angle은 직교좌표계에 익숙한 사람들에게 직관적으로 이해되지만 단점이 많아서 단점을 잘 이해하고 써야합니다. Euler angle은 유명한 짐벌락 문제(gimbal lock problem, [영상](https://www.youtube.com/watch?v=zc8b2Jo7mno))이 있어서 각도 범위에 제한이 있습니다.

Euler angle을 이용해 SE(3) 변환을 표현한다면 다음과 같은 6차원 벡터가 될 것입니다.


$$
\mathbf{p} = \begin{bmatrix} t_x & t_y & t_z & \alpha & \beta & \gamma \end{bmatrix}^T
$$



### 5.4 Quaternion

4개의 숫자로 표현할 수 있는 quaternion은 일종의 복소수이며 스칼라 $$q_0$$와 벡터 $$\mathbf{q}$$로 이루어졌습니다. 상황에 따라 간단히 4차원 벡터로 표현하기도 합니다. 


$$
q = q_0 + \mathbf{q} = q_0 + q_1\mathbf{i} + q_2\mathbf{j} + q_3\mathbf{k} \\
q = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}
$$


Quaternion으로 회전을 표현할 때는 크기가 1인 **unit quaternion**만 사용할 수 있습니다. ($$\begin{vmatrix} q\end{vmatrix}=1$$)  Unit quaternion에서 스칼라 $$q_0$$는 각도를 의미하고 벡터 $$\mathbf{q}$$는 회전축을 의미합니다. SO(3)에 속하는 임의의 회전변환은 특정 축을 기준으로 특정 각도를 회전시켜 만들수 있습니다. Euler angle은 직교 좌표축으로만 회전했기 때문에 세 번 회전해야 임의의 회전을 만들어낼 수 있지만 임의의 회전축을 사용할 수 있다면 한번의 회전으로 모든 회전을 구현할 수 있습니다.  

어떤 회전이 회전축 $$\mathbf{v}$$를 기준으로 $$\theta$$ 만큼 회전해야 한다면 quaternion으로 다음과 같이 표현할 수 있습니다.  


$$
q = cos{\theta \over 2} + \mathbf{v}sin{\theta \over 2} \\
q = \begin{bmatrix} cos{\theta \over 2} \\ v_x sin{\theta \over 2} \\ 
v_y sin{\theta \over 2} \\ v_z sin{\theta \over 2} \end{bmatrix}
$$


Quaternion으로 실제로 어떤 좌표계를 회전시켜야 한다면 quaternion product를 계산해야 합니다. Quaternion 사이의 곱셈에 해당하는 quaternion product는 다음과 같이 계산합니다.


$$
pq = p_0q_0 - \mathbf{p \cdot q} + p_0\mathbf{q} + q_0\mathbf{p} + \mathbf{p} \times \mathbf{q} \\
pq= 
\begin{bmatrix} p_0 & -p_1 & -p_2 & -p_3 \\ 
p_1 & p_0 & -p_3 & p_2 \\ p_2 & p_3 & p_0 & -p_1 \\ 
p_3 & -p_2 & p_1 & p_0 \end{bmatrix}
\begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}
= 
\begin{bmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ 
q_1 & q_0 & q_3 & -q_2 \\ q_2 & -q_3 & q_0 & q_1 \\ 
q_3 & q_2 & -q_1 & q_0 \end{bmatrix}
\begin{bmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \end{bmatrix}
$$


어떤 3차원 벡터 $$\mathbf{p}$$를 $$q$$라는 quaternion으로 회전된 좌표계에서 본 $$\mathbf{p}'$$ 를 계산하는 식은 다음과 같습니다. Quaterion product를 앞 뒤로 두 번하는 것입니다.


$$
\bar p' = q^* \bar p q \\
\bar p = \begin{bmatrix} \mathbf{p} \\ 0 \end{bmatrix}, \quad 
q^*= q_0 - \mathbf{q}
$$


이러한 quaternion 연산이 부담스럽다면 rotation matrix로 변환할수도 있습니다.


$$
\bar p' = q^* \bar p q \\
\mathbf{p}' = Q \mathbf{p} \\
Q = \begin{bmatrix} 
2q_0^2-1 + 2q_1^2 & 2q_1q_2 - 2q_0q_3 & 2q_1q_3 + 2q_0q_2 \\
2q_1q_2 + 2q_0q_3 & 2q_0^2-1 + 2q_2^2 & 2q_2q_3+2q_0q_1 \\
2q_1q_3 - 2q_0q_2 & 2q_2q_3 + 2q_0q_1 & 2q_0^2-1 + 2q_2^3
\end{bmatrix}
$$


반대로 rotation matrix를 quaternion으로 변환하고 싶다면 임의의 rotation matrix에서 회전 축 $$\mathbf{v}$$와 회전 각도 $$\theta$$를 알아내면 됩니다.


$$
R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ 
r_{21} & r_{22} & r_{23} \\ 
r_{31} & r_{32} & r_{33} \end{bmatrix} \\
\theta = acos \begin{pmatrix}{Tr(R) - 1 \over 2} \end{pmatrix} \\
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}
= {1 \over 2sin\theta} \begin{bmatrix} r_{23} - r_{32} \\ r_{31} - r_{13} \\ r_{12} - r_{21} \end{bmatrix} \\
q = cos{\theta \over 2} + \mathbf{v}sin{\theta \over 2}
$$


Quaternion으로 회전을 표현할때의 장점은 다음과 같습니다.

- 짐벌락 문제와 같은 표현 자체의 결점이 없음
- 임의의 회전을 회전축과 회전각을 이용해 직관적으로 표현 가능
- 9개의 숫자로 표현되고 복잡한 조건을 가진 rotation matrix에 비해 quaternion은 4차원 벡터로 표현할 수 있고 조건이 단순해서 (unit quaternion) 경제적이고 최적화에도 유리함
- rotation matrix를 거치지 않고 quaternion product를 통해 연속적인 회전을 직접 연산가능,   $$R_3 = R_1 R_2 \leftrightarrow q_3 = q_1 q_2$$



Quaternion을 이용해 SE(3) 변환을 표현한다면 이동 벡터와 합친 7차원 벡터가 됩니다.


$$
\mathbf{p} = \begin{bmatrix} t_x & t_y & t_z & q_w & q_x & q_y & q_z \end{bmatrix}^T
$$



### 5.5 Twist Coordinates

*Twist*의 의미를 이해하기 위해서는 어려운 용어들과 미분방정식을 푸는 유도과정을 봐야하지만 여기선 생략하고 결과적인 사용방법만 알아보겠습니다. ORB-SLAM이나 LSD-SLAM 등 대부분의 유명한 VO/SLAM 논문에서는 이 방법으로 회전을 표현합니다.

앞서 quaternion이 euler angle의 문제를 해결했는데 왜 또 다른걸 배워야 할까요? Quaternion도 사용해보면 완벽하게 편하진 않기 때문입니다. Rotation matrix보다는 단순하지만 **unit** quaternion이라는 조건도 걸려있고 원래 3자유도의 회전을 4차원 벡터로 표현하는 것도 아쉬운 점입니다. 앞서 임의의 rotation matrix를 회전각도 $$t$$와 회전축 $$\omega$$로 해석할 수 있다고 했습니다.
$$
R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ 
r_{21} & r_{22} & r_{23} \\ 
r_{31} & r_{32} & r_{33} \end{bmatrix} \\
t = acos \begin{pmatrix}{Tr(R) - 1 \over 2} \end{pmatrix} \\
\mathbf{\omega} = \begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{bmatrix}
= {1 \over 2sin(t)} \begin{bmatrix} r_{23} - r_{32} \\ r_{31} - r_{13} \\ r_{12} - r_{21} \end{bmatrix} \\
$$


반대로, 회전각도 $$t$$와 회전축 $$\omega$$로부터 rotation matrix R을 만들어낼 수 있는 공식이 있습니다. 바로 로드리게스 공식입니다.

> **Rodrigues' formula for rotation matrix**: Given $$\omega \in \mathbb{R}^3$$ with $$\begin{Vmatrix} \omega \end{Vmatrix}=1$$ and $$t \in \mathbb{R}^3$$, the matrix exponential $$R = e^{\hat \omega t}$$ is given by the following formula:


$$
e^{\hat \omega t} = I + \hat \omega sin(t) + {\hat \omega}^2(1 - cos(t)) \\
\hat \omega = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0 \end{bmatrix}
$$


$$\hat \omega$$은 **twist** 혹은 **exponential coordinate**이라 불리며 임의의 3차원 벡터 $$\omega$$로부터 만들어질 수 있는 $$\hat \omega$$의 집합을 $$so(3)$$라고 합니다. 자세히 보면 $$\hat \omega$$의 모양이 벡터 $$\omega$$와의 cross product에 대한 행렬연산이라는 것을 알 수 있습니다.


$$
\mathbf{u} \times \mathbf{v} = 
\begin{bmatrix} 0 & -u_3 & u_2 \\ u_3 & 0 & -u_1 \\
-u_2 & u_1 & 0 \end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}
= \hat{\mathbf{u}} \mathbf{v}
$$


임의의 twist $$\hat{\mathbf{u}} = \hat \omega t$$는 **exponential map**을 통해 rotation matrix가 될 수 있고 exponential map을 구현한 식이 로드리게스 공식입니다.


$$
exp : so(3) \to SO(3) \\
\hat{\mathbf{u}} \in so(3) \mapsto R = e^{\hat{\mathbf{u}}} \in SO(3)
$$


다시 정리해보면 세 단계의 표현 방식이 있습니다.

| term                                 | expression                                                   |
| ------------------------------------ | ------------------------------------------------------------ |
| twist coordinates                    | $$\mathbf{u} = \omega t \in \mathbb{R}^3$$                   |
| twist or <br>exponential coordinates | $$\hat{\mathbf{u}} = \begin{bmatrix} 0 & -u_3 & u_2 \\ u_3 & 0 & -u_1 \\ -u_2 & u_1 & 0 \end{bmatrix} \in so(3)$$ |
| rotation matrix                      | $$R = e^{\hat{\mathbf{u}}} \in SO(3)$$                       |

세 가지 표현 모두 동일한 정보를 가지고 있으므로 아무런 제약조건이 없는 3차원 벡터인 "twist coordinates"로 임의의 회전을 표현할 수 있다는 뜻이 됩니다. 이는 앞서 배운 rotation matrix, euler angle, quaternion의 모든 단점들이 해소된 표현 방법이라고 볼 수 있습니다. 3차원으로 표현이 compact하고 아무런 제약조건도 없습니다.

벡터의 표현도 직관적으로 이해할 수 있습니다. Twist coordinate $$\mathbf{u} = \omega t$$ 에서 크기 $$t = \begin{Vmatrix} \mathbf{u} \end{Vmatrix}$$는 각도를 의미하고 방향 $$\omega = {\mathbf{u}  \over \begin{Vmatrix} \mathbf{u} \end{Vmatrix}}$$ 은 회전축을 의미합니다.  

다만 주의할점은 각도가 $$2\pi$$마다 반복되므로 twist coordinates와 rotation matrix는 one-to-one 관계가 아니라 many-to-one 관계라는 것입니다.


$$
R = e^{\hat{\omega} t} = e^{\hat{\omega} (t+2\pi n)}
$$


위에서 본 세 단계의 표현은 이동(translation)을 더한 rigid transformation에도 그대로 적용됩니다. 이동을 $$\mathbf{v} \in \mathbb{R}^3$$로 표현할 때 이동을 포함한 twist coordinates와 twist는 다음과 같이 표현합니다.


$$
\xi = \begin{bmatrix} \mathbf{v} \\ \mathbf{u} \end{bmatrix} \in \mathbb{R}^6 \\
\hat{\xi} = \begin{bmatrix} \hat{\mathbf{u}} & \mathbf{v} \\ 0 & 1 \end{bmatrix} \in se(3)
$$


이로부터 transformation matrix를 구하는 Rodrigues' formula는 다음과 같습니다.


$$
e^{\widehat{\xi}t} = 
\begin{bmatrix} e^{\widehat{w}} & 
\left( I - e^{\widehat{w}} \right) \widehat{w} v 
+ w \widehat{w}^T v \\
0 & 1
\end{bmatrix}
= \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \in SE(3)
$$

---
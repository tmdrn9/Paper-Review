## Abstract

전통적인 3D 워터마킹 접근법/상용 소프트웨어는 일반적으로 3D 메쉬에 메시지를 내장하고 나중에 왜곡/왜곡되지 않은 워터마킹된 3D 메쉬에서 메시지를 직접 검색하도록 설계

↓  많은 경우 사용자는 3D 메쉬 대신 렌더링된 2D 이미지에만 액세스 가능. 

새로운 end-to-end framework 소개

1. mesh geometry과 textures모두에 메시지를 은밀하게 내장하는 인코더
2. 다양한 카메라 각도와 다양한 조명 조건에서 워터마킹된 3D 객체를 렌더링하는 차별화 가능한 렌더러
3. 2D 렌더링된 이미지에서 메시지를 복구하는 디코더

실험을 통해 우리의 모델이 인간에게 시각적으로 감지할 수 없는 정보를 내장하고, 3D 왜곡을 겪는 2D 렌더링에서 내장된 정보를 검색하는 방법을 배울 수 있음을 보여줌

광선 추적기 및 실시간 렌더링기와 같은 다른 렌더링기와도 미세 조정 없이 작동할 수 있음을 보여줌

> 렌더링이란? 
3차원 공간에 객체(Object)를 2차원 화면인 하나의 장면(Scene)에 바꾸어 표현하는 것
> 

## Introduction

key contributions

- 렌더링된 2D 이미지에서 3D mesh로 인코딩된 메시지를 검색하고 3D 워터마킹 사용을 넓힐 수 있는 최초의 3D to 2D 워터마킹 방법 제시
- 미분 가능한 렌더링을 사용함으로써 제안한 방법을 완전히 차별화할 수 있으며, 이를 통해 전체 프레임워크를 차별화 가능한 3D 왜곡 모음으로  end-to-end 훈련 가능
- 디코더는 미분 불가능한 렌더러로부터 내장된 메시지를 디코딩할 수 있으며, 미세 조정을 통해 개선 가능성 o

## Method

크게 encoder, distortion, differentiable renderer, 그리고 decoder 모듈로 구성

![pipeline](https://github.com/tmdrn9/Paper-Review/assets/77779116/0ca1b04c-6a65-47ce-9f92-045234cfa20a)


### Definitions

[mesh $\mathbb{M}(V,F,T,P)$]

- $V \in \mathbb{R}^{N_{v}\times C_{v}}$ : 꼭짓점.
    - $N_{v}$는 꼭짓점 개수, $C_{v}$는 3D 위치/색상/2D텍스터 좌표요소(?)/normal? 등등
- $F \in {\{0,... , N_{v}-1\}}^{N_{f}\times C_{f}}$: mesh faces
    - V의 인덱스를 포함. $N_{f}$는 면의 수, $C_{f}$는 mesh의 꼭짓점 개수(ex 삼각형 메쉬의 경우 3)
- $T \in \mathbb{R}^{H_{t}\times W_{t} \times C_{t}}$: texture information
- $P \in \mathbb{R}^{10}$ : material color information
    - ambient, diffuse, specular RGB color, and a shininess 포함

[binary message]

- $M \in \{0,1\}^{N_b}$ :  $N_b$는 메세지 길이

### Encoder

$F$를 변경하면 바람직하지 않은 아티팩트가 생성되고 정보 숨기기에 $P$가 너무 작아서 불가능 → $V, T$를 변경→ vertex 위치를 변경하면 퇴화된 삼각형이 생성+backpropagation불가능 ⇒normal, texture요소에만 인코딩(둘 다 또는 둘 중 하)

각 메세지 비트를 $N_v$회 복제하여  $N_v \times N_b$ tensor 구성→ $V$요소와 concat

⇒Def. $V_m\in \mathbb{R}^{N_{v}\times (C_{v}+N_b)}$

각 메세지 비트를 $H_t \times W_t$회 복제하여  $H_t \times W_t \times N_b$ tensor 구성→$T$와 concat

⇒Def. $T_m\in \mathbb{R}^{H_{t}\times W_{t} \times (C_{t}+N_b)}$

watermarked $V=V_e=E_G(V_m)$

- shape은 original $V$와 동일
- $E_G$: PointNet 아키텍쳐를 fully-convolutional로 만들어 사용

watermarked $T=T_e=E_T(V_m)$

- shape은 original $T$와 동일
- $E_T$: Hidden와 유사한 CNN기반 아키텍쳐 사용

### Distortions

워터마킹 시스템이 다양한 왜곡에 견고하게 만들기 위한 module.

1. Gaussian noise: 평균 µ, 분산 σ
2. Random axis-angle rotation: α, x, y, z
3. Random scaling: s
4. Random cropping on 3D vertices

*Since all these distortions are differentiable, we could train our network end-to-end. 뭔말?

### Differentiable Rendering

2D 렌더링된 이미지에서 메시지를 추출할 수 있는 3D 워터마킹 시스템을 훈련하기 위해서는 차별화 가능한 렌더링 계층이 필요

Genova et al. [[Unsupervised training for 3d morphable model regression](https://openaccess.thecvf.com/content_cvpr_2018/papers/Genova_Unsupervised_Training_for_CVPR_2018_paper.pdf)]를 3step으로 따름

1. Rasterization: 픽셀당 화면 공간 버퍼와 삼각형 내부 픽셀의 무게 중심 좌표를 계산
    - https://github.com/google/tf_mesh_renderer
2. deferred shading
    - Phong reflection model 사용
3. Splatting: 래스터화된 각 표면 점이 픽셀 중심에 있는 스플랫으로 변환되고 해당 음영 색상에 의해 채색

이 접근 방식은 픽셀 그리드에서 메쉬 Vertice와 vertex별 속성에 대해 매끄러운 미분 계산.

제안된 파이프라인은 $L_I^i$강도를 가진 $L_P^i$ 에 위치한 고정된 점광원이 있다고 가정

조명 매개변수 $L$, 카메라 행렬$K$, 그리고 메쉬들 $\mathbb{M}$이 주어지면 출력 2D 이미지를 생성하기 위해 우리의 차별화 가능한 렌더러 사용

$$
I=R_D(\mathbb{M},K,L)\in \mathbb{R}^{H_r\times W_r\times 3}
$$

![example](https://github.com/tmdrn9/Paper-Review/assets/77779116/427abfee-d028-416d-8fa2-43f6823cb979)


### Decoder

렌더링된 이미지 $I$로부터 메세지를 추출하기 위해 네트워크 $D$ 사용 ⇒ $M_r=D(I)$

network D

- 가변 크기 입력 이미지를 허용하기 위해 여러 개의 conv layer후에 하나의 global pooling layer 사용한 후 여러 개의 fully-connected layers
- 마지막 fully-connected layer는 메세지 길이인 고정된 개수의 출력 노드를 갖게 설정

상용 소프트웨어?와 같은 미분 불가능한 렌더링에서 디코더 성능을 더욱 향상시키기 위해, 우리는 특정 렌더링된 출력으로 디코더를 미세 조정하거나 단독으로 훈련 가능

최종 메시지는 아래 식처럼 binarization 진행

$$
M_{br}=clamp(sign(M_r-0.5),0,1)
$$

### Losses

1. vertex watermarking loss:  $V_e$와 $V$사이의 거리 계산.

$$
L_{vertex}(V,V_{e})=\sum_{i} w^{i}(\frac{\sum_{\alpha}|V^{i}[\alpha]-V^{i}_{e}[\alpha]|}{N_vC_v})
$$

&ensp;&ensp;  * $w_i$: 각 구성 요소에 대한 가중치로 각 정점 구성 요소에 대한 변화의 민감도 조정 가능.


2. texture watermarking loss:  $T_e$와 $T$사이의 거리 계산.
    
$$
L_{texture}(T,T_{e})=\frac{\sum_{\alpha}|T[\alpha]-T_{e}[\alpha]|}{H_tW_tC_t}
$$
    
3. 2D rendering loss: 원본 메쉬의 렌더링된 이미지 $I_o$와 워터마킹된 메쉬의 렌더링된 이미지 $I_w$의 차이 계산.
    
$$
L_{image}(I_o,I_w)=\frac{\sum_{\alpha}|I_o[\alpha]-I_w[\alpha]|}{3H_wW_w}
$$
    
4. message loss: $M_{rb}$와 원본 $M$ 차이 계산. 디코딩 오류에 불이익을 부여.
    
$$
L_{message}(M,M_r)=\frac{\sum_{\alpha}|M[\alpha]-M_r[\alpha]|}{N_b}
$$
    
$\therefore$ Full loss

$$
L_{total}=λL_{vertex}+γL_{texture}+δL_{image}+θL_{message}+ ηL_{reg}
$$

&ensp;&ensp;  *$L_{reg}$는 regularization loss

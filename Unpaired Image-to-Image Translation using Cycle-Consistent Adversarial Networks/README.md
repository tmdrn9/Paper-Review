velog : [click here!](https://velog.io/@victory/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-NetworksCycleGAN)

***

# Introduction


#### 제안배경 
![paper's figure](https://images.velog.io/images/victory/post/a4c9e851-c374-42d6-8260-bda22a01f328/image.png)

수년간 컴퓨터 비전, 이미지 처리, 컴퓨터 사진 및 그래픽에 대한 연구는 지도학습 즉 보시는 그림같이 pair 데이터셋(쌍을 이루는 데이터셋) 환경에서 강력한 변환시스템을 만들었습니다.
하지만 pair데이터 셋을 얻는 것은 어렵고 비쌉니다. 
또한 객체 변환과 같은 많은 작업의 경우 출력이 제대로 정의되지 않습니다.

그래서 저자들은 <span style="color:red">**Unpaired dataset만으로 도메인 간 번역을 할 수 있는 알고리즘**</span>을 찾은 것입니다.

#### 제안한 알고리즘

도메인 사이에 기본 관계가 있다고 가정합니다.
![made by victory](https://images.velog.io/images/victory/post/34a5943c-7686-4b4d-a9b7-a95384c12976/image.png)
일반적으로 그림의 화살표 왼쪽같이 G를 X를 Y에 매핑하는 Generator라고하면 Y와 일치하는 Y^에대한 출력 분포를 유도할 수 있습니다.

하지만 동일한 분포를 유도하는 무수히 많은 Generator가 있기에 이러한 변환은 개별적인 X와 Y^이 의미 있는 방식으로 짝을 이룬다는 것을 보장하지 않습니다. 뿐만 아니라 적대적인 목표를 단독적으로 최적화하는 것도 어렵습니다. (종종 모든 입력 이미지가 동일한 출력 이미지에 매핑 되고 최적화가 진전을 이루지 못하는 잘 알려진 모드 축소 문제를 초래)

그래서 저자는 <span style="color:red">**주기 일관성 특성**</span>을 이용하기로 합니다. 주기 일관성을 예를 들어 설명하자면 우리가 특정 문장을 영어에서 프랑스어로 번역하고, 그것을 다시 영어로 번역하면 원래의 문장으로 돌아와야 한다는 것입니다.
**즉, X를 Y로 매핑하는 Generator G뿐만아니라 Y를X로 매핑하는 Generator F도 사용해 유의미하게 매핑 되는 Generator를 만드는 것입니다.**

여기서 두개의 Generator는 일대일 대응이 되어야합니다.

![made by victory](https://images.velog.io/images/victory/post/fc917797-11aa-4e57-b301-4a5d687e11eb/image.png)

위 그림의 구조가 Cycle GAN입니다.

학습 한번 할때 Dataloader에서 X, Y를 뱉어내면 X는 Y에 매핑되게 Generator G를 훈련시키고 Y는 X에 매핑되게 Generator F를 각각 훈련시킵니다.
그 후 각각의 output을 각각의 Discriminator DY와 DX에 넣어 적대적 손실을 계산하고 주기 일관성 손실을 계산하기위해 앞서 훈련시킨 Generator G와F를 가져와 X와 Y를 재구성해 주기일관성 손실을 계산합니다.

혼동이 올 수도 있는데 그림 상 위, 아래에 있는 X^, Y^은 다른 것입니다.(혼란을 방지하기 위해 색깔로 표시해뒀습니다)
위에서의 Y^은 X를 Y처럼 만든것이고, X^은 그 Y^을 다시 원래 X처럼 만든 것입니다. 
마찬가지로 아래서의 X^은 Y를 가지고 X처럼 만들고자한 것이고, Y^은 그 X^을 가지고 다시 Y로 되돌리고자 한 것으로 아예 다른 것입니다.
***
# Formulation

**총 손실은 적대적 손실과 주기일관성 손실을 더한 것입니다.**

#### Adversarial Loss
![Adversaril loss made by victory](https://images.velog.io/images/victory/post/4fa9baba-bd99-4a85-9036-22eb76d44934/image.png)
우선 Generator와 Discriminator가 각각 두 개로 적대적 손실도 두 개가 나옵니다.
그림의 왼쪽 구조와 공식으로 설명하겠습니다

**Discriminator의 목표는 진짜인지 가짜인지 잘 구별하는 것이고 Generator의 목표는 가짜를 진짜같이 만드는 것입니다.**

우선 왼쪽항은 G,DY,X,Y로 구성된 함수 LGAN이 있을 때 G는 LGAN의 값을 낮추려하고 DY는 LGAN의 값을 크게한다는 뜻입니다.
오른쪽항을 보시면 우선 Generator는 두번째항에만 관여합니다. 
Generator가 목표대로 잘 작동한다면 Discriminator가 진짜라고 속아 output이 1에 가깝게 나와 항이 작아져 LGAN값이 작아질 것입니다.
반대로 Discriminator가 목표대로 잘 작동한다면 Discriminator에 Y를 넣으면 Discriminator의 output이 1에 수렴하고 Y^을 넣었을 때는 0에 수렴해 LGAN값이 최대로 높일 것입니다.

이게 식의 전부입니다. 
오른쪽도 변수명만 다르지 동일한 식입니다.
생각보다 쉽죠?😆

#### Cycle Consistency Loss

![Cycle Consistency Loss made by victory](https://images.velog.io/images/victory/post/e04f10e5-8bf7-4d89-8881-98a0d3254cd3/image.png)
Generator G와F를 사용해서 변환한 것을 다시 변환해 재구성한 것과 원본을 비교해 주기 일관성 손실을 계산합니다.
논문에서는 L1norm을 이용해 원본과 재구성한 것의 차이를 계산하였습니다

#### Full Objective

![Full Objective made by victory](https://images.velog.io/images/victory/post/2bff3aff-c298-4b07-98e4-3bc34c14c393/image.png)

정리하자면 두 개의 Generator의 output들을 Discriminator에 입력으로 해 각각의 손실을 얻고, 주기일관성 손실을 얻기위해 앞에 학습한 서로의 Generator를 가져옵니다. 그후 다시 원본 분포에 가깝게 만든 output들로 주기일관성 손실을 계산하는 것입니다

추가적으로 논문에서는 실험부분에서 identity loss도 사용합니다. 이는 입력 데이터의 색상을 보존하기 위한 손실입니다. X를 Y로 매핑하는 Generator G에 입력으로 Y를 넣어 얻은 output Y^과 Y와의 거리(L1norm), Y를 X로 매핑하는 Generator F에 X를 넣어 얻은 output X^과 X와의 거리(L1norm)를 더한 것입니다. 
이것은 Generator 학습 시 손실에 더해줍니다.

***
# Experimental Results

#### 실험1 - 모델별 성능 평가
![paper's figure](https://images.velog.io/images/victory/post/3c256120-41f4-4086-8579-801d6a20f766/image.png)
이 실험은 여러 모델로 변환작업을 해본 것입니다
오른쪽 사진을 보나 표를 보나 CycleGAN이 주목할만한 결과를 얻지는 못합니다. 
하지만 종종 지도학습인 Pix2Pix와 유사한 품질로 변환이 되었다는 것을 주목해서 보시면 됩니다.

#### 실험2 - Loss Function
![paper's figure](https://images.velog.io/images/victory/post/5385e681-dd40-43b5-aac3-887d71fbe9bc/image.png)
Cycle alone은 주기일관성 손실만 사용했을 때, GAN alone은 적대적손실만 사용했을 때, 세-네번째는 적대적 손실과 양방향 주기 일관성 손실이 아니라 각각 한 방향의 주기일관성 손실을 쓸때이고, 마지막이 저자가 제안한 주기일관성 손실과 적대적손실을 다 사용했을때입니다.

맨위 적대적 손실이랑 주기일관성 각각만 사용했을때 결과가 크게 저하된것을 볼수있는데 이로써 둘개 모두가 결과에 중요하다는 것을 알 수 있습니다.
또한 주기일관성 손실을 하나씩만 쓰면 종종 훈련 불안정성을 야기하고 모드 붕괴를 유발하며, 특히 제거된 매핑 방향에 대해 모드 붕괴를 유발한다는 것을 발견했다고 합니다.

#### 실험3 - 이미지 재구성 품질
![paper's figure](https://images.velog.io/images/victory/post/fe132153-8544-4c7b-a62e-0dafbfeab40f/image.png)
그림은 재구성된 이미지의 몇 가지 무작위 샘플 입니다.
사진,지도 예시와 같이 하나의 도메인이 훨씬 다양한 정보를 나타내는 경우에도 재구성된 영상이 원본 X에 가까운 경우가 많다는 것을 관찰되었습니다.

#### 응용
![paper's figure](https://images.velog.io/images/victory/post/6886a0c8-6e87-46e3-88d1-945fe918ccf3/image.png)
사진을 여러 화풍으로 바꾸거나 오른쪽 그림처럼 그림을 사진처럼 바꾸기도합니다. 

![paper's figure](https://images.velog.io/images/victory/post/6669f94a-84a2-4fa8-a235-b7d96a752af2/image.png)
또한 말을 얼룩말로 변환을 하거나, 겨울을 여름으로 혹은 여름을 겨울로 바꾸거나, 오렌지,사과를 양방향으로 바꾸고, 그냥 사진을 아웃오브포커스사진으로 바꾸는 등 많은 응용이 가능합니다.

***
# Limitations

![paper's figure](https://images.velog.io/images/victory/post/f4e95bef-38c3-4568-9a68-7ddaa51422b6/image.png)

첫번째, 결과가 균일하게 잘 나오지 않는다는 것입니다. 색상, 질감 변경을 포함하는 변환작업에서는 종종 성공하나 기하학적 변화가 필요한 작업은 거의 성공하지 못 합니다. (ex - 고양이를 개로 변환하는 작업) 저자는 이러한 기하학적 변화를 다루는 것은 앞으로의 풀어 가야할 중요한 문제라고 언급합니다.

두번째로는 training dataset의 특성 분포에서 야기되는 문제입니다. 예를 들어 설명하면 말을 얼룩말로 변환하는 모델을 학습시킬 때 training dataset에 사람이 들어가지 않고 학습하게 되면 사람이 포함되어 있는 사진으로 test를 한다면 사람도 얼룩말의 패턴을 가지게 변환됩니다.

마지막으로는 쌍으로 구성된 훈련 데이터로 달성할 수 있는 결과와 쌍으로 구성되지 않은 방법에 의해 달성된 결과 사이의 차이가 여전하다는 것입니다. 어떤 경우에는 이 차이를 좁히기 매우 어렵거나 심지어 불가능할 수 있다고 합니다.

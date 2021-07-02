## [DACON] 북극 해빙예측 AI 경진대회

- 대회 주소: https://dacon.io/competitions/official/235731/overview/description
- 주제 및 배경
지구 온난화가 진행되면서 지구 표면 온도 상승, 이상 기온, 해수면 상승 등 다양한 기후 변화가 관측되고 있으며, 북극 해빙 또한 지구 온난화의 영향으로 매년 면적이 줄어들고 있다. 과거 관측된 해빙 데이터를 통해 앞으로 변화할 해빙을 예측해보자.

- 대회 설명
1978년부터 관측된 북극 해빙의 변화를 이용하여 주별 해빙 면적 변화 예측

- 전략
	- 대회측에서 공개한 baseline에서는 **ConvLSTM**모델 사용
	- 2021년 1월에 공개된 논문 **Spatio-temporal Weather Forecasting and Attention Mechanism on Convolutional LSTMs** 에서는 ConvLSTM 보다 좋은 성능을 낸 모델을 소개함.
	- 위 논문에서 제시한 모델을 사용하여 대회 참여

---

### 논문소개

논문: [http://arxiv.org/abs/2102.00696](http://arxiv.org/abs/2102.00696)
코드: https://github.com/sftekin/ieee_weather

- 기상 데이터 & 관측값 모두 사용 → 고해상도 수치 기상 데이터 예측하는 딥러닝 아키텍쳐 제안
- input series의 다른 부분에 초점을 맞춘 attention matrices
	- 공간적 및 시간적 상관 관계를 캡처하는데 있어 상당한 개선을 보여줌.
	- 정성적 및 정량적 결과 제공

- 시공간적 예측으로 문제 해결
- model 구성
	- Convolutional Long-short Term Memory
	- Convolutional Neural Network units with encoder-decoder structure
	- attention and a context matcher mechanism을 통해 단기성능과 해석 가능성 향상
	- attention weights and ConvLSTM units를 함께 훈련

- 과정
	1. 모델 → 시공간 상관관계(spatial-temporal correlations) 학습
	2. 예측 제공 ← interpretability(해석 가능성)이 높은 multiple spatio-temporal series(다중 시공간)를 사용
	3. flow vectors → input spatiotemporal series의 고유한 특성 제공
	4. spatial and temporal resolution(공간 및 시간 해상도)가 높은 실제 dataset로 네트워크 성능 분석


#### The Weather Model

A spatio-temporal forecasting model for the Numerical Weather Prediction.



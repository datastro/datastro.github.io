---
layout: post
title: "뇌졸중 예측 모델링(3/3) - SMOTE와 언더샘플링"
date: 2026-07-19 00:00:00 +0900
categories: [데이터·분석]
tags: [데이터분석, 머신러닝, 불균형데이터]
---

2부에서 threshold를 0.028까지 낮춰 recall 0.80을 확보했지만, precision은 0.07까지 떨어졌다. 뇌졸중 아닌 사람을 뇌졸중 위험군으로 지목하는 경우가 실제 뇌졸중 환자보다 12배 많아지는 수준이었다. 그렇다면 threshold를 건드리는 대신, 애초에 학습 데이터의 클래스 비율 자체를 맞추면 이 트레이드오프가 완화되지 않을까. 이 글에서는 SMOTE와 언더샘플링을 실제로 적용해서 비교해봤다.

## 세 가지 방식을 같은 테스트셋에서 비교한다

비교 대상은 세 가지다. 원본 비율을 그대로 쓰는 Imbalanced, 소수 클래스를 다수 클래스 수만큼 복제·보간하는 SMOTE, 다수 클래스를 소수 클래스 수만큼 줄이는 언더샘플링이다. 셋 다 학습 데이터에만 적용하고, 평가는 항상 원본 비율(2.89%)을 유지한 동일한 테스트셋에서 수행했다. 이 부분이 핵심이다. 리샘플링한 데이터로 평가까지 해버리면 실제 데이터 생성 환경과 다른 비율에서 성능을 측정하기에 의미가 없어진다.

```python
from imblearn.over_sampling import SMOTE

def get_smote_split(df_input, list_features, str_target='stroke', float_test_size=0.25, int_random_state=42):
    X_train, X_test, y_train, y_test = get_stratified_split_data(
        df_input, list_features, str_target, float_test_size, int_random_state
    )
    obj_smote = SMOTE(random_state=int_random_state)
    X_train_res, y_train_res = obj_smote.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test

def get_balanced_split(df_input, list_features, str_target='stroke', float_test_size=0.25, int_random_state=42):
    df_minority = df_input[df_input[str_target] == 1]
    df_majority = df_input[df_input[str_target] == 0].sample(n=len(df_minority), random_state=int_random_state)
    df_balanced = pd.concat([df_minority, df_majority])
    return get_stratified_split_data(df_balanced, list_features, str_target, float_test_size, int_random_state)
```

Imbalanced 학습셋은 정상 22,819 / 뇌졸중 680(2.89%)이고, SMOTE는 22,819 / 22,819(1:1), 언더샘플링은 680 / 680(1:1)이다. 다섯 개 모델을 세 가지 방식으로 각각 학습시키고, threshold는 모두 기본값 0.5로 고정한 채 같은 테스트셋에서 Recall·Precision·F1·AUC를 측정했다.

## Recall은 확실히 오른다

![샘플링 전략별 recall AUC F1 비교, Imbalanced SMOTE Undersampling 3패널 바 차트](/assets/images/posts/stroke-prediction-3-sampling-comparison.jpg)

recall만 보면 효과가 뚜렷하다. 로지스틱 회귀는 Imbalanced에서 Recall 0.000이었던 게 SMOTE 0.705, 언더샘플링 0.789로 오른다. 랜덤포레스트는 0.000 → 0.692 → 0.855, XGBoost는 0.009 → 0.670 → 0.841로 비슷한 패턴이다. Threshold를 전혀 건드리지 않고 기본값 0.5를 그대로 뒀는데도 Recall이 이만큼 오른다는 것이 SMOTE·언더샘플링의 직관적인 매력이다.

그런데 AUC를 같이 보면 얘기가 달라진다. Imbalanced 대비 SMOTE의 AUC는 Logistic Regression 0.815→0.775, RandomForest 0.810→0.789, XGBoost 0.793→0.751, GradientBoosting 0.811→0.781, NaiveBayes 0.780→0.751로 다섯 모델 전부 낮아졌다. 언더샘플링은 반대로 0.815→0.817, 0.810→0.837, 0.793→0.825처럼 대체로 소폭 상승했다.

AUC는 Threshold와 무관하게 모델이 양성과 음성을 얼마나 잘 순위 매기는지를 보는 지표다. SMOTE가 Recall은 올렸는데 AUC를 오히려 깎았다는 건, 모델이 뇌졸중 사례를 더 잘 구별하게 된 게 아니라는 뜻이다. 그렇다면 Recall 상승의 정체는 뭘까.

## Recall이 오른 진짜 이유: 확률 자체가 밀려 올라간다

같은 로지스틱 모델이 실제 뇌졸중 사례에 대해 내놓는 예측 확률의 분포를 세 가지 방식별로 그려봤다.

![샘플링 방식에 따른 예측 확률 분포 변화, 실제 뇌졸중 발생 사례 기준 로지스틱 회귀](/assets/images/posts/stroke-prediction-3-probability-shift.jpg)

Imbalanced로 학습한 모델은 실제 뇌졸중 환자에게도 낮은 확률(0.0~0.15 부근)을 주는 경우가 많다. 반면 SMOTE와 언더샘플링으로 학습한 모델은 같은 사람들에게 훨씬 높은 확률을 준다. 학습 데이터를 1:1로 맞추는 순간 모델이 학습하는 사전 확률(prior) 자체가 2.89%에서 50%로 바뀌기 때문에, 예측 확률 전체가 오른쪽으로 밀려 올라간다. threshold 0.5는 이제 원래 데이터 기준으로 보면 훨씬 낮은 지점에 해당하게 되고, 그래서 Recall이 저절로 오른 것처럼 보인다.

즉 SMOTE·언더샘플링으로 얻은 Recall 상승은 2부에서 Threshold를 낮춰 Recall을 올렸던 것과 원리상 같은 종류의 조작이다. 다만 Threshold를 직접 낮추는 대신, 학습 데이터의 사전 확률을 바꿔서 간접적으로 같은 효과를 낸 것뿐이다.

## 확률값 자체는 신뢰할 수 있는가: Calibration Curve

여기서 진짜 문제가 드러난다. 사전 확률이 밀려 올라간 모델이 내놓는 "확률"이라는 숫자를 실제 위험도로 해석해도 되는지를 확인해야 한다. Calibration curve는 모델이 예측한 확률 구간별로 실제 관측된 양성 비율이 얼마나 일치하는지를 보여준다.

```python
from sklearn.calibration import calibration_curve

for str_regime in ['Imbalanced', 'SMOTE', 'Undersampling']:
    array_proba = dict_proba_store[(str_regime, 'Logistic')]
    array_frac_pos, array_mean_pred = calibration_curve(
        y_true=y_test_common, y_prob=array_proba, n_bins=10, strategy='quantile'
    )
```

![샘플링 방식에 따른 예측 확률 보정 상태 비교, calibration curve 로지스틱 회귀](/assets/images/posts/stroke-prediction-3-calibration-curve.jpg)

완벽하게 보정된 모델이라면 대각선(y=x) 위에 점들이 놓여야 한다. 세 방식 모두 대각선보다 한참 아래에 있지만, 특히 SMOTE와 언더샘플링 쪽이 더 두드러진다. 모델이 "위험도 0.85"라고 예측한 구간에서 실제 뇌졸중 발생 비율은 0.12 정도에 그친다. 반면 Imbalanced 모델은 예측 확률이 애초에 0~0.12 구간을 벗어나지 않는다. 절대적인 보정 상태는 세 모델 다 이상적이지 않지만, SMOTE와 언더샘플링은 실제 발생 확률보다 6~7배 높은 숫자를 위험도로 제시하고 있다는 게 확인된다.

이것이 SMOTE·언더샘플링을 그대로 배포할 때 생기는 문제라고 볼 수 있다. Recall이 올라간 건 사실이지만, 그 대가로 모델이 내놓는 확률값의 절대적인 의미가 깨진다. "이 사람의 뇌졸중 위험도는 85%다"라는 출력을 실사용자에게 보여준다면, 실제로는 그 사람이 뇌졸중일 확률이 12% 남짓이라는 걸 모델도 사용자도 모르는 채로 지나가게 된다.

## 현장에 적용한다면: 확률값을 그대로 보여줄 수 있는가

이 계열의 모델을 현장에 놓는다고 하면 보통 두 가지 방식 중 하나다. 하나는 2부처럼 Threshold로 양성/음성만 나누는 것이고, 다른 하나는 확률값 자체를 "위험도"로 사용자나 임상의에게 보여주는 것이다. 후자를 택한다면 이번 Calibration Curve 결과가 곧바로 Feasibility의 문제가 된다. SMOTE·언더샘플링으로 학습한 모델이 내놓는 0.85라는 숫자를 그대로 화면에 띄우면, 실제 발생률(0.12)의 7배에 달하는 위험도를 사용자에게 전달하는 셈이다. 이런 숫자는 불필요한 불안을 유발하거나, 반대로 몇 번 반복되면 "이 앱이 말하는 위험도는 원래 부풀려져 있다"는 학습 효과로 사용자가 경고 자체를 무시하게 만들 수도 있다. 둘 다 스크리닝 도구로서는 치명적이다.

이 문제를 풀려면 재보정(Recalibration) 절차가 필요하다. 학습 시 인위적으로 맞춘 사전 오즈(1:1)와 원본 사전 오즈(0.0289:0.9711)의 차이를 로지스틱 모델의 절편에서 빼주는 방식으로 어느 정도 되돌릴 수 있다. 다만 이 프로젝트에서는 애초에 원본 비율을 유지한 Imbalanced 모델을 최종안으로 택했기 때문에 이 재보정 절차 자체가 필요 없었다. Imbalanced 모델의 확률값은 절대적으로 완벽하게 보정돼 있지는 않지만(대각선에서 벗어나 있는 건 마찬가지다), 적어도 예측 확률의 스케일 자체는 원본 유병률과 같은 자릿수(0~0.15 부근)에 머물러 있어서, SMOTE·언더샘플링 모델보다는 왜곡의 정도가 훨씬 작다.

그리고 1부에서 짚은 한계, 즉 이 데이터가 전향적으로 추적한 코호트가 아니라 단면조사라는 점도 여기서 다시 걸린다. 재보정을 아무리 잘 해도, 이 모델의 확률값이 뜻하는 건 "설문 시점에 이미 뇌졸중을 진단받은 사람과의 통계적 연관성"이지 "앞으로 몇 년 안에 뇌졸중이 발생할 확률"이 아니다. 두 가지는 수치상으로 비슷해 보여도 실제 의미는 다르다. 그러니 Calibration이 완벽해진다고 해도, 이 모델을 그대로 현장에 놓기 전에 전향적 코호트에서 재현되는지를 확인하는 절차가 하나 더 필요하다. 이번 시리즈에서 다룬 Recall·Precision·Calibration은 전부 이 데이터셋 내부에서의 타당성이고, 현장 적용 가능성은 그 바깥의 검증까지 거쳐야 완성된다.

## 그렇다면 SMOTE·언더샘플링은 왜 존재하는가

처음 던졌던 질문은 "현실은 불균형 데이터인데, 그 비율과 다르게 학습하는 리샘플링 기법이 스코어를 올리는 목적 외에 효용이 있는가"였다. 이번 실험 결과를 놓고 보면 두 가지로 나눠서 답할 수 있을 것 같다.

리샘플링이 실제로 기여하는 부분은 있다. 언더샘플링은 AUC까지 소폭 개선시켰는데, 이는 소수 클래스 표본(680건)이 학습 과정에서 다수 클래스에 완전히 묻히지 않고 결정 경계에 충분히 반영되도록 만들어주기 때문으로 보인다. 극단적으로 표본이 적은 소수 클래스에서는 리샘플링 없이 학습이 아예 소수 클래스의 패턴을 놓치는 경우도 있는데, 이런 상황에서는 리샘플링이 실질적인 신호 확보 수단이 된다. 계산 효율 측면에서도 언더샘플링은 학습 데이터 크기를 680:680으로 줄여주기 때문에, 대규모 데이터에서 반복 실험 속도를 높이는 실용적 이점이 있다.

반면 이번 실험에서 SMOTE가 보여준 것처럼, AUC 하락을 감수하면서 Recall만 끌어올리는 효과는 Threshold 조정과 본질적으로 다르지 않다. 그리고 어느 쪽이든, 원본 비율과 다른 사전 확률로 학습한 모델을 그대로 배포하면 확률값 자체가 실제 위험도보다 부풀려진 채로 나간다. 리샘플링한 모델을 실사용에 쓰려면, 학습 시 바뀐 사전 오즈와 원본 사전 오즈의 차이를 절편(intercept)에서 보정하거나, Platt scaling 같은 재보정 절차를 거쳐야 확률값이 의미를 되찾는다.

이번 프로젝트에서 원본 비율을 그대로 유지하는 쪽을 택한 이유가 여기에 있다. 재보정 절차를 추가하지 않고도 확률값이 실제 유병률과 정합적으로 유지되고, 필요한 Recall 수준은 2부에서처럼 Threshold 이동으로 직접 통제할 수 있어서, 리샘플링에 따르는 추가적인 보정 부담을 애초에 지지 않아도 됐다.

## 시리즈를 마치며

1부에서는 국민건강영양조사 데이터로 뇌졸중 예측 모델을 만들고 AUC 0.78~0.82를 확인했다. 2부에서는 Threshold 0.5에서 Recall이 0에 가까웠던 문제를 Threshold 이동으로 Recall 0.80까지 끌어올렸고, 그 대가로 Precision이 0.07까지 떨어지는 트레이드오프를 확인했다. 3부에서는 SMOTE와 언더샘플링이 Recall을 쉽게 올려주지만, 그 상승이 Threshold 이동과 같은 원리(사전 확률 이동)에서 온다는 것과, 그로 인해 예측 확률의 보정 상태가 깨진다는 것을 확인했다.

세 편을 관통하는 결론은 하나다. Recall이든 샘플링이든, 숫자 하나를 올리는 조작 자체는 정형화되어 있지만 그 조작이 모델의 다른 속성(Precision, Calibration)에 어떤 대가를 요구하는지를 확인하지 않으면 모델링이 완결되지 않는다는 것이다. 그리고 그 대가가 실제 현장에서 감당할 만한지는 이 데이터셋 안에서는 끝내 답할 수 없었다. 2부에서 본 정상인 12.8명당 환자 1명이라는 트레이드오프도, 3부에서 본 위험도 7배 과대추정 문제도, 그것이 괜찮은지 아닌지는 검진 현장의 Capacity와 비용 구조, 그리고 전향적 코호트에서의 재현 여부에 달려 있다. 이번 시리즈가 낸 결론은 그 질문에 대한 답이 아니라, 그 질문을 숫자로 구체화하는 것까지다.

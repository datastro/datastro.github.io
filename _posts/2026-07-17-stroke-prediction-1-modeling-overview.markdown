---
layout: post
title: "국민건강영양조사로 뇌졸중 예측 모델링 해보기 (1) - 불균형 데이터와 변수 선정"
date: 2026-07-17 01:10:00 +0900
categories: [데이터·분석]
tags: [데이터분석, 머신러닝, 불균형데이터, 국민건강영양조사]
---

국민건강영양조사(HN) 데이터를 가지고 뇌졸중 발생 여부를 예측하는 모델링 팀 프로젝트를 진행했다. 학회 발표까지 마쳤지만 두 가지 의문은 계속 남았다. 하나는 "불균형 데이터에서 Recall만 쫓는 게 정말 정답인가"였고, 다른 하나는 "SMOTE나 언더샘플링은 결국 현실의 비율과 다른 데이터로 학습하는 건데, 스코어를 올리는 것 말고 다른 효용이 있을까"였다. 이 두 의문을 정리하려고 글을 세 편으로 나눠 쓴다. 1부는 모델링 자체를 소개하고, 2부는 Recall 우선 전략에 대한 생각, 3부는 샘플링 전략에 대한 생각을 다룰 예정이다.

## 데이터 개요

국민건강영양조사 2014년~2021년 자료를 통합해서 사용했다. 성별, 연령, 결혼상태, 소득, 교육, 직업 같은 인구통계 변수와 흡연·음주·운동 같은 건강행태 변수, 고혈압·당뇨·고지혈증·심근경색 등 기저질환 진단 여부 변수까지 총 44개 변수를 뇌졸중(stroke) 진단 여부와 함께 정리했다. 결측치가 있는 행은 모두 제거했고, 무응답 코드(88, 99 등)가 섞인 범주형 응답도 제거했다.

최종적으로 남은 표본은 31,333명이고, 이 중 뇌졸중 진단을 받은 사람은 907명이다. 비율로는 2.89%다.

![국민건강영양조사 표본의 뇌졸중 진단 여부 분포, 정상 97.11% 뇌졸중 진단 2.89%](/assets/images/posts/stroke-prediction-1-target-distribution.jpg)

97대 3 정도의 불균형이다. 이 정도면 흔히 말하는 "레어 이벤트" 예측 문제에 들어간다. 이후 시리즈 전체에서 다루는 Recall 논쟁과 샘플링 논쟁도 결국 이 2.89%라는 숫자에서 시작한다.

## 이 데이터가 실제 현장과 다른 점

모델링으로 들어가기 전에 짚어야 할 한계가 있다. 국민건강영양조사는 병원이나 검진기관에 실제로 들어오는 사람들의 흐름, 즉 현장 데이터가 아니다. 매년 전국에서 표본을 추출해서 진행하는 단면조사(cross-sectional survey)다. '뇌졸중' 변수도 영상 판독이나 진료 기록이 아니라 "의사에게 뇌졸중을 진단받은 적이 있는지"를 묻는 설문 문항에 대한 **자가보고**다. 그리고 조사 시점에 이미 진단을 받은 사람과 아닌 사람을 나눈 것이지, 앞으로 몇 년 안에 뇌졸중이 발생할 사람을 전향적으로 추적한 게 아니다.

이 두 가지 차이가 이 시리즈 전체에서 다루는 Feasibility(현장 적용 가능성) 논의의 출발점이다. 이 모델이 학습한 건 "현재 뇌졸중 진단자와 비진단자를 가르는 패턴"이지, "미래에 뇌졸중이 생길 사람을 미리 짚어내는 패턴"이 아니다. 스크리닝의 실제 목적이 후자라는 걸 감안하면, 이번 모델링은 그 목적에 대한 완전한 답이라기보다 방법론을 연습해보는 시뮬레이션에 더 가깝다. 실제 스크리닝 도구로 쓰려면 전향적으로 추적한 코호트 데이터로 별도 검증하는 과정이 반드시 필요하다. 이 한계를 지금 정리하면 좋을것 같다.

## 변수 선정: Cramér's V

44개 변수를 모두 넣고 모델링할 수도 있지만, 범주형 변수 간 연관성을 먼저 확인하고 상위 변수만 추리는 쪽을 택했다. 범주형-범주형 변수 사이의 연관성은 카이제곱 통계량 기반의 Cramér's V로 측정했다.

```python
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

def calculate_cramers_v(series_x, series_y):
    df_crosstab = pd.crosstab(series_x, series_y)
    float_chi2 = chi2_contingency(df_crosstab)[0]
    int_n = df_crosstab.sum().sum()
    int_r, int_k = df_crosstab.shape
    if min(int_r - 1, int_k - 1) == 0:
        return 0.0
    return np.sqrt((float_chi2 / int_n) / min(int_r - 1, int_k - 1))

list_rows = []
for str_col in list_candidate_cols:
    float_v = calculate_cramers_v(df_final[str_col], df_final['stroke'])
    list_rows.append((str_col, float_v))

df_corr_rank = pd.DataFrame(list_rows, columns=['variable', 'cramers_v'])
df_corr_rank = df_corr_rank.sort_values(by='cramers_v', ascending=False)
```

상위 변수를 뽑아보면 연령대(0.160)와 고혈압 진단 여부(0.157)가 가장 강하게 연관되어 있고, 교육상태(0.120), 소득상태(0.111), 당뇨(0.105), 직업(0.102), 심근경색(0.088), 고지혈증(0.083) 순으로 이어진다.

![뇌졸중과 연관성이 높은 변수 상위 15개, Cramer's V 기준 랭킹 바 차트](/assets/images/posts/stroke-prediction-1-cramers-v-ranking.jpg)

연령과 고혈압이 가장 앞에 오는 건 임상적으로도 자연스럽다. 다만 순위 자체보다 중요한 건, 이 상위 10개 변수를 골라내는 과정에서 이미 "무엇을 모델에 넣을지"에 대한 판단이 개입된다는 점이다. Cramér's V 기준으로 자른 상위 10개가 실제로 예측력이 가장 좋은 조합이라는 보장은 없다. 다만 변수가 44개에서 10개로 줄어들면 이후의 모델 해석에서 용이하기 때문에, 실무적 타협으로 상위 10개를 최종 변수로 확정했다.

이 상위 10개 변수끼리도 서로 독립적이지는 않다. 대각선 위쪽을 가려서 중복을 없앤 상태로 변수 간 Cramér's V를 다시 계산해봤다.

![상위 10개 변수 간 Cramer's V 상관관계, 대각선 및 우상단을 가린 삼각형 히트맵](/assets/images/posts/stroke-prediction-1-cramers-v-pairwise-heatmap.jpg)

연령대와 고혈압이 0.46으로 가장 높고, 교육상태와 고지혈증이 0.34, 소득상태와 직업이 0.33, 교육상태와 소득상태가 0.31로 뒤를 잇는다. 즉 뇌졸중과의 연관성 기준으로 뽑은 상위 10개 변수 중 상당수가 서로 겹치는 정보를 담고 있다는 뜻이다. 연령대가 높을수록 고혈압 진단율이 자연히 올라가고, 소득·교육·직업처럼 사회경제적 지표들끼리도 서로 얽혀 있다. 이런 다중공선성은 이후 로지스틱 회귀의 오즈비를 해석할 때 각 변수의 효과를 완전히 독립적인 기여로 읽으면 안 된다는 것을 뜻한다. 예를 들어 고혈압의 오즈비에는 연령의 효과 일부가 섞여 들어가 있을 가능성이 있다.

## 베이스라인 모델링: 원본 비율 그대로

모델링에서 가장 먼저 정한 원칙은 "학습 데이터의 클래스 비율을 건드리지 않는다"였다. 클래스 가중치(class_weight)도 주지 않고, SMOTE나 언더샘플링도 쓰지 않고, 실제 관측된 2.89% 비율을 그대로 유지한 채 학습·평가했다. 왜 이러한 학습 방법을 선택했는지는 3부에서 다룰 예정이다.

```python
from sklearn.model_selection import train_test_split

def get_stratified_split_data(df_input, list_features, str_target, float_test_size=0.25, int_random_state=42):
    df_encoded = pd.get_dummies(df_input[list_features], drop_first=True).astype(int)
    series_y = df_input[str_target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df_encoded, series_y,
        test_size=float_test_size,
        stratify=series_y,
        random_state=int_random_state
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_stratified_split_data(
    df_input=df_final,
    list_features=list_top_vars,
    str_target='stroke'
)
# Train 분포 - 0: 22819, 1: 680 (2.89%)
# Test 분포  - 0: 7607,  1: 227  (2.89%)
```

층화추출(stratify)을 써서 train/test 모두 원본과 동일한 2.89% 비율을 유지하게 했다. 이 상태로 Logistic Regression, RandomForest, XGBoost, GradientBoosting, NaiveBayes 다섯 개 모델을 학습시키고 ROC-AUC로 비교했다.

![베이스라인 모델 ROC 곡선 비교, 원본 비율 유지 가중치 없음](/assets/images/posts/stroke-prediction-1-roc-comparison.jpg)

AUC는 Logistic Regression 0.815, GradientBoosting 0.811, RandomForest 0.810, XGBoost 0.793, NaiveBayes 0.780으로 다섯 모델 모두 0.78~0.82 사이에 몰려 있다. 모델 종류에 따른 차이보다는 변수 자체의 정보량이 성능을 결정하는 구간에 가깝다는 뜻으로 읽었다.

숫자만 보면 꽤 쓸만한 모델처럼 보인다. 그런데 이 AUC 뒤에 숨어있는 문제가 있다. threshold를 기본값 0.5로 두고 실제로 예측을 시켜보면 얘기가 달라진다.

![베이스라인 모델 혼동행렬 비교, threshold 0.5 기준 원본 비율 유지](/assets/images/posts/stroke-prediction-1-confusion-matrix-default.jpg)

Logistic Regression과 RandomForest는 뇌졸중 사례 227건을 전부 음성으로 분류해서 Recall이 정확히 0.000이다. GradientBoosting도 227건 중 단 1건도 잡아내지 못했다. XGBoost는 227건 중 2건만 양성으로 잡아 recall 0.009에 그쳤다. 다섯 모델 중 유일하게 NaiveBayes만 146건을 잡아내 recall 0.643을 기록했는데, 대신 정상인 7,607명 중 1,771명을 양성으로 분류해서 Precision은 0.076밖에 안 된다. AUC 0.78~0.82라는 숫자만으로는 이 차이가 전혀 보이지 않는다.

이런 결과가 나오는 이유는 모델이 부실해서가 아니다. 학습 데이터에서 뇌졸중 비율이 2.89%밖에 안 되기 때문에, 모든 사람을 그냥 "정상"이라고 찍어도 전체 정확도는 97%를 넘는다. Threshold 0.5는 두 클래스가 나올 확률이 비슷할 때 의미가 있는 기준점인데, 유병률이 3%에도 못 미치는 이 데이터에는 애초에 맞지 않는 기준이다. 그 결과 Logistic Regression·RandomForest·GradientBoosting은 뇌졸중 클래스에 0.5를 넘는 확률을 거의 배정하지 않았고, XGBoost도 극소수(2건)에만 넘겼다. NaiveBayes만 Recall이 높게 나온 건 역설적으로 각 변수를 독립으로 가정하는 단순한 구조 탓에 확률 추정이 다른 모델보다 극단적으로 흔들리기 때문이지, 더 잘 학습해서가 아니다. 이 문제를 Threshold 조정으로 어떻게 풀 수 있는지, 그리고 그렇게 얻은 Recall이 어떤 대가를 요구하는지는 2부에서 다룬다.

## 로지스틱 회귀로 본 위험 요인

AUC나 Recall과는 별개로, 어떤 변수가 뇌졸중 위험을 얼마나 높이는지도 확인해봤다. 이진형 변수만 골라 로지스틱 회귀를 적합하고 오즈비(odds ratio)를 계산했다.

```python
import statsmodels.api as sm

X_dum = sm.add_constant(pd.get_dummies(df_logistic.drop(columns=['stroke']), drop_first=True).astype(int))
y_logit = df_logistic['stroke']

model_logit = sm.Logit(y_logit, X_dum).fit(disp=0)

df_coef = pd.DataFrame({
    '변수': model_logit.params.index,
    'coef': model_logit.params.values,
    'p_value': model_logit.pvalues.values
})
df_coef['odds_ratio'] = np.exp(df_coef['coef'])
```

![주요 변수에 따른 뇌졸중 발병 오즈비, 고혈압 3.51배 심근경색 3.07배 70대 이상 2.61배 당뇨 1.67배 고지혈증 1.31배](/assets/images/posts/stroke-prediction-1-odds-ratio.jpg)

고혈압 진단을 받은 경우 뇌졸중 오즈가 3.51배, 심근경색 진단 이력이 있으면 3.07배, 70대 이상이면 2.61배, 당뇨는 1.67배, 고지혈증은 1.31배 높았다. 모두 p<0.001 수준에서 유의했다. 고혈압과 심근경색이 상위에 오는 건 두 질환 모두 혈관계 손상을 공유하는 만큼 자연스러운 결과다.

## 정리

여기까지가 모델링의 뼈대다. 국민건강영양조사 통합 데이터에서 31,333명 중 뇌졸중 진단자는 907명(2.89%)이었고, Cramér's V로 상위 10개 변수를 골라 원본 비율을 유지한 채 다섯 개 모델을 학습시켰다. AUC는 0.78~0.82로 준수했지만, 기본 threshold(0.5)에서는 대부분의 모델이 뇌졸중 사례를 거의 잡아내지 못했다. 오즈비 분석에서는 고혈압과 심근경색, 고령이 가장 강한 위험 요인으로 나타났다.

다음 글에서는 이 recall=0 문제를 어떻게 다뤘는지, 그리고 "Threshold를 낮춰서 Recall을 올린다"는 접근이 정말 이 문제의 답이 될 수 있는지를 다룬다.

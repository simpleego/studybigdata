# LOF(LOcal Outlier Forest)
> 이상치 탐지 기법 중 하나인 **LOF (Local Outlier Factor)**는 밀도 기반의 방법으로,
> 데이터 포인트가 주변 이웃들과 얼마나 다른지를 측정해 이상값을 판단합니다. 

---

## 🔍 LOF (Local Outlier Factor)란?

LOF는 **데이터 포인트의 지역 밀도(local density)**를 기준으로 이상값을 판단합니다.  
즉, 어떤 점이 주변 이웃들보다 밀도가 낮으면 **이상값일 가능성이 높다**고 판단합니다.

### 📌 핵심 개념

- **k-이웃(k-nearest neighbors)**: 각 점의 주변 이웃을 k개 선택
- **Local Reachability Density (LRD)**: 이웃들과의 거리 기반으로 밀도 계산
- **LOF 점수**: 이웃들의 밀도와 해당 점의 밀도를 비교한 비율

> LOF 점수가 **1에 가까우면 정상**,  
> **1보다 크면 이상값 가능성 있음**,  
> **값이 클수록 더 이상치일 가능성 높음**

---

## 🧪 예시: 2D 데이터에서 LOF 적용

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

# 예시 데이터 생성
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers += 2

X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))

X = np.concatenate([X_inliers, X_outliers], axis=0)

# LOF 모델 적용
lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title("LOF 이상치 탐지 결과")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
```

### 🔍 결과 해석
<img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/31835fdc-ca67-429d-9917-e32b2688be41" />


- `y_pred = -1`인 점들은 LOF에 의해 **이상값으로 판단된 점들**
- 시각화에서는 보통 빨간색 점들이 이상값으로 표시됨
- LOF 점수가 높을수록 주변 밀도와 차이가 크다는 의미

---

## ✅ LOF의 장점

- **비선형 분포**에서도 잘 작동
- **지역적 밀도 차이**를 고려하므로, 전체 분포에 의존하지 않음
- **고차원 데이터**에도 적용 가능

## ⚠️ 주의할 점

- `n_neighbors` 값에 따라 결과가 달라질 수 있음
- 밀도가 균일하지 않은 데이터에서는 민감할 수 있음

---

## 📐 LOF 공식 (LaTeX 마크다운)

### ✅ LOF 점수 계산

<img width="889" height="304" alt="image" src="https://github.com/user-attachments/assets/0b6af52a-e5c8-4308-83d0-fa5acf0056a7" />

### ✅ Reachability Distance

<img width="650" height="60" alt="image" src="https://github.com/user-attachments/assets/1558aba2-c895-41ac-896f-910f6ee6ef50" />

- 즉, 너무 가까운 이웃은 k-distance로 보정하여 거리 계산

---

### ✅ Local Reachability Density (LRD)

<img width="449" height="92" alt="image" src="https://github.com/user-attachments/assets/88d4576b-206a-404b-9208-102a8ce251aa" />


- 이웃들의 LRD 평균을 𝑝의 LRD로 나눈 값
- LOF가 1보다 크면 주변보다 밀도가 낮다는 의미 → 이상값 가능성 높음

---

## 📊 해석

| LOF 점수 | 해석 |
|----------|------|
| ≈ 1      | 정상값 (이웃과 밀도 유사) |
| > 1      | 이상값 가능성 있음 (이웃보다 밀도 낮음) |
| ≫ 1      | 강한 이상값 (고립된 위치) |

---

이 수식은 LOF의 핵심 아이디어인 **이웃과 비교한 상대적 밀도**를 수학적으로 표현한 것

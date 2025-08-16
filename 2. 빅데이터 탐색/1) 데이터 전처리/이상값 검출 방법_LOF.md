좋아요! 이상치 탐지 기법 중 하나인 **LOF (Local Outlier Factor)**는 밀도 기반의 방법으로, 데이터 포인트가 주변 이웃들과 얼마나 다른지를 측정해 이상값을 판단합니다. 아래에 LOF의 개념과 예시를 통해 쉽게 설명해볼게요.

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

```markdown
$$
\text{LOF}_k(p) = \frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \frac{\text{LRD}_k(o)}{\text{LRD}_k(p)}
$$
```

- \( p \): 대상 데이터 포인트  
- \( N_k(p) \): \( p \)의 k-최근접 이웃  
- \( \text{LRD}_k(p) \): \( p \)의 Local Reachability Density

---

### ✅ Reachability Distance

```markdown
$$
\text{reach\_dist}_k(p, o) = \max(\text{k-distance}(o), \text{dist}(p, o))
$$
```

- 너무 가까운 이웃은 k-distance로 보정

---

### ✅ Local Reachability Density (LRD)

```markdown
$$
\text{LRD}_k(p) = \left( \frac{1}{\frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \text{reach\_dist}_k(p, o)} \right)
$$
```

- 이웃들과의 평균 도달 거리의 역수  
- 밀도가 높을수록 LRD 값이 큼

---

## 💡 마크다운에서 수식 깨짐 방지 팁

- **블로그나 노트 앱**에서는 `$$ ... $$`로 감싸면 블록 수식으로 잘 보입니다.
- **GitHub README**에서는 LaTeX 수식이 기본적으로 지원되지 않으므로 이미지로 대체하거나 HTML로 변환해야 합니다.
- **Jupyter Notebook**이나 **Obsidian**, **Typora** 같은 마크다운 지원 툴에서는 위 방식이 잘 작동합니다.

---

혹시 사용하는 플랫폼이나 툴이 있다면, 거기에 맞게 수식이 잘 보이도록 도와줄게요!

---

## 📊 해석

| LOF 점수 | 해석 |
|----------|------|
| ≈ 1      | 정상값 (이웃과 밀도 유사) |
| > 1      | 이상값 가능성 있음 (이웃보다 밀도 낮음) |
| ≫ 1      | 강한 이상값 (고립된 위치) |

---

이 수식은 LOF의 핵심 아이디어인 **"이웃과 비교한 상대적 밀도"**를 수학적으로 표현한 것

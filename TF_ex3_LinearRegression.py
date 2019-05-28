'''
Regression(회귀)
- 어떤 연속형 데이터 Y와 이 Y의 원인이 되는 X간의 관계를 추정하기 위해 만든 형식의 관계식
- 하지만 실제 데이터 측정에는 여러가지 원인으로 인하여 데이터의 손실이 발생
- 따라서 정확한 관계식을 만들 수 없음
- 그러므로 회귀 모델링을 통한 오차의 합이 최소가 되도록 만들어 주려함

Linear Regression
- 한개의 종속변수(Dependent Variable)와 설명변수(Explanatory Variables)들과의 관계를 모델링
- 관계를 정의하기 위해 방정식 사용
'''

# H(x)=Wx+b

import numpy as np

num_points=1000
vectors_set=[]

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]

import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()
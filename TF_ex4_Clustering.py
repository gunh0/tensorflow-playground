'''
Clustering
- 선형 회구 모델에는 레이블이 있는 것은 아니다.
- 분석을 위한 Clustering(군집화)가 필요하다.
- 군집화를 위한 k-means 알고리즘을 사용하여 데이터를 다른 묶음과 구분되도록 유사한 것끼리 자동으로 그룹화한다.
- K-평균 알고리즘(K-means algorithm)은 주어진 데이터를 k개의 클러스터로 묶는 알고리즘으로,
  각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작한다.
- K-means 알고리즘에서는 추정할 목표 변수나 결과 변수가 없다.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(0.0, 0.9)])

df = pd.DataFrame({"x": [v[0] for v in vectors_set], "y": [v[1] for v in vectors_set]})
print(df)
sns.lmplot(x="x", y="y", data=df, fit_reg=False, size=6)
plt.show()

vectors = tf.constant(vectors_set)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k-1, 0]))
#

print(centroides)
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centrodies = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.math.subtract(expanded_vectors, expanded_centrodies)), 2), 0)
# 2 AttributeError: module 'tensorflow' has no attribute 'sub'
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                     reduction_indices=[1]) for c in range(k)])

update_centrodies = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess=tf.Session
sess.run(init_op)

for step in range(100):
    _, centroides_values, assignments_values = sess.run([update_centrodies, centroides, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignments_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignments_values[i])

df =  pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

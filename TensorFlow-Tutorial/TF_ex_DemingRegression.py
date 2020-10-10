#!/usr/bin/env python
# coding: utf-8

# Deming Regression은 Total Regression(전회귀)로도 불린다.
#
# Deming Regression는 y값과 x값의 오차를 최소화한다.
#
# Deming Regression을 구현하기 위해서는 Loss Cost Function을 수정해야 한다.
#
# 일반적인 선형 회귀의 비용함수는 수직거리를 최소화하기 때문이다.
#
# 직선의 기울기와 y절편을 이용하여 점까지 수식 거리를 구하고 tensorflow가 그 값을 최소화 하게 된다.

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ops.reset_default_graph()

iris = load_iris()
print(iris.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])


print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# load the data
x_val = iris.data[:, 3]  # petal width
y_val = iris.data[:, 0]  # sepal length


# initialize placeholders
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_uniform(shape=[1, 1]))


# In[2]:


with tf.Session() as sess:
    fomula = tf.add(tf.matmul(x_data, A), b)
    demm_numer = tf.abs(tf.subtract(fomula, y_target))  # numerator
    demm_denom = tf.sqrt(tf.add(tf.square(A), 1))  # denominator
    loss = tf.reduce_mean(tf.truediv(demm_numer, demm_denom))  # 점과 직선사이의 거리

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.15)
    train_step = opt.minimize(loss)

    init = tf.global_variables_initializer()
    init.run()

    loss_vec = []
    batch_size = 125

    for i in range(1000):
        rand_idx = np.random.choice(len(x_val), size=batch_size)
        rand_x = x_val[rand_idx].reshape(-1, 1)
        rand_y = y_val[rand_idx].reshape(-1, 1)

        my_dict = {x_data: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict=my_dict)
        temp_loss = sess.run(loss, feed_dict=my_dict)
        loss_vec.append(temp_loss)

        if (i+1) % 100 == 0:
            print('step {}: A={}, b={}, Loss={}'.format(
                i+1, A.eval(), b.eval(), temp_loss))

    [slope] = A.eval()
    [cept] = b.eval()


# In[4]:


best_fit = []
for i in x_val.ravel():
    poly = i*slope[0] + cept[0]
    best_fit.append(poly)

_, axes = plt.subplots(1, 2)
axes[0].scatter(x_val, y_val, edgecolors='k', label='Data Points')
axes[0].plot(x_val, best_fit, c='red', label='Best fit line')
axes[0].set_title('Petal Width vs Sepal Length', size=12)
axes[0].set_xlabel('Petal Width')
axes[0].set_ylabel('Sepal Length')
axes[0].legend(loc=2)

axes[1].plot(loss_vec, c='k')
axes[1].set_title('Demming Loss per Generation', size=12)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Demming Loss')

plt.show()

import tensorflow as tf

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
# Placeholder는 선언 후 그 다음 값을 전달한다.
# 따라서 반드시 데이터가 제동되어야 한다.
# Float32는 큰 용량의 데이터를 사용할 때 GPU메모리 또는 속도 등의 문제에 주로 사용한다.

c=tf.add(a,b)

sess=tf.Session()
print(sess.run(c,feed_dict={a : 3, b : 4}))
print(sess.run(c,feed_dict={a : [1, 3], b : [4, 6]}))


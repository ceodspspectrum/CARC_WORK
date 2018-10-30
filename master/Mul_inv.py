import numpy as np
import tensorflow as tf
import timeitMultiplication
import pandas as pd
t_1=[]
t_2=[]
N=20
s=2
l=1
input1= tf.placeholder(tf.float32)
#input2 = tf.placeholder(tf.float32)
output = tf.linalg.inv(input1)
res=pd.DataFrame(columns=('N','Time'))
num_repeats=10
for i in range(0,N):

    A=np.random.rand(l,l)
    #B=np.random.rand(l,l)

    with tf.Session() as sess:

        timer = timeit.Timer("sess.run([output],feed_dict={input1:A})", setup="import tensorflow  as tf; from __main__ import sess, A,  output,input1")
        tensorflow_times_list = timer.repeat(num_repeats, 1)

    res.loc[i]=[l,np.mean(tensorflow_times_list)]
    res.to_csv('Result_gpu_inv.csv',index=False)
    l=l*s

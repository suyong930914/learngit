import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#设置tensorflow/numpy随机种子
tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('D:\Python深度学习\人工智能第5个月\深度学习\本月发的\DeepLearning_tensorflow_code\MNIST_DATA',one_hot=True)

#设置超参数
learning_rate = 0.01
training_items = 10000
batch_size = 128    #每次训练图片
n_input = 28    #MNIST data input img shape(28*28)
n_steps = 28    #time steps 时间步伐
n_hidden_unis = 128     #neurons hidden layer 隐藏神经元
n_classes = 10

#图像占位符输入
X = tf.placeholder(tf.float32,[None,n_steps,n_input])   #shape(batch,784)
Y = tf.placeholder(tf.float32,[None,n_classes])
#define weight定义权重
weights = {'in':tf.Variable(tf.random_normal(shape=[n_input,n_hidden_unis])),
           'out':tf.Variable(tf.random_normal(shape=[n_hidden_unis,n_classes]))}
#define biase 定义偏执项
biases = {'in':tf.Variable(tf.constant(value=0.1,shape=[n_hidden_unis,])),
          'out':tf.Variable(tf.constant(value=0.1,shape=[n_classes,]))}
#定义RNN模型函数
def RNN(X,weights,biases):
    # 隐藏层 hidden
    X = tf.reshape(X,shape=[-1,n_input])
    X_in = tf.matmul(X,weights['in']) + biases['in']
    X_in = tf.reshape(X_in,shape=[-1,n_steps,n_hidden_unis,])
    #细胞层 cell
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_unis,    #隐藏层的数
                                            forget_bias=1.0,    #忘记初始值
                                            state_is_tuple=True)    #生成的state是否为元组
    initial_state = rnn_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(
        cell = rnn_cell,    #cell you have chosen所选单元
        inputs = X_in,      #input输入图像
        initial_state = initial_state,  #the initial hidden state最初隐藏状态
        time_major = False,     #False:(batch,time_step,input)
    )
    #输出结果
    result = tf.matmul(states[1],weights['out']) + biases['out']
    return result

pred = RNN(X,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for i in range(200):
        batch_X,batch_Y = mnist.train.next_batch(batch_size)
        batch_X = batch_X.reshape([batch_size,n_steps,n_input])
        _,cost_val = sess.run([train_op,cost],feed_dict={X:batch_X,Y:batch_Y})
        if i % 20 == 0:
            print('accuracy:',sess.run(accuracy,feed_dict={X:batch_X,Y:batch_Y}))
            print('cost:',cost_val)

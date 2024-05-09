import tensorflow as tf
import numpy as np
import time
import math
import cifar-10-data
max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir='cifar_data/cifar-10-batches-bin'
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
        tf.add_to_collection('losses',weights_loss)
    return var
images_train,labels_train=cifar-10-data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
image_test,labels_test=cifar-10-data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.float32,[batch_size])
kerenl1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1=tf.nn.conv2d(x,kerenl1,[1,1,1,1],padding='same')
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='same')
kerenl2=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(x,kerenl2,[1,1,1,1],padding='same')
bias2=tf.Variable(tf.constant(0.0,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='same')
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)
weight3=variable_with_weight_loss(shape=[192,10],stddev=0.04,w1=0.004)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.nn.relu(tf.matmul(local4,weight3)+fc_bias3)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result)
weights_with_l2_loss=tf.add_n(tf.get_collection('losses'))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op=tf.nn.in_top_k(result,y_,1)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()
    for step in range(max_steps):
        start_time=time.time()
        image_batch,labels_batch=sess.run([images_train,labels_train])
        _,loss_values=sess.run([train_op,loss],feed_dict={x:image_batch,y_:labels_batch})
        duration=time.time()-start_time
        if step%100==0:
            examples_per_sec=batch_size/duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (step, loss_values, examples_per_sec, sec_per_batch))
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))
    true_count=0
    total_sample_count=num_batch*batch_size
    for j in range(batch_size):
        image_batch,label_batch=sess.run([image_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count+=np.sum(predictions)
        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))



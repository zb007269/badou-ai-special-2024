from nets import vgg16
import numpy as np
import tensorflow as tf
import utils
img1 =utils.load_shape('./test_data/dog.jpg')
inputs =tf.placeholder(img1,tf.float32,[None,None,3])
resized_img =utils.resize_img(inputs,(224,224))
predition=vgg16.vgg_16(resized_img)
ckpt_filename='./model/vgg_16.ckpt'
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver =tf.train.Saver()
saver.restore(img1,ckpt_filename)
pro =tf.nn.softmax(predition)
pre =sess.run(pro,feed_dict={inputs:img1})
print('result:')
utils.print_prob(pre[0],'./synset.txt')

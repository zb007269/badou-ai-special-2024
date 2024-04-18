import cv2
import tensorflow as tf
import matplotlib.image as mping
import numpy as np
from tensorflow.python.ops import array_ops
def load_image(path):
    img=mping.imread(path)
    short_edge= min(img.shape)
    xx=int((img.shape[1]-short_edge)/2)
    yy = int((img.shape[0] - short_edge) / 2)
    crop_img =img[xx:xx+short_edge,yy:yy+short_edge]
    return crop_img
def resize_image(image,size):
    with tf.name_scope('resize_size'):
        images=[]
        for i in image:
            i=cv2.resize(i,size)
            images.append(i)
        images=np.array(images)
        return images

def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
        synset=[l.split(',')[1][:-1] for l in f.readlines()]
        print(synset[argmax])
        return synset[argmax]





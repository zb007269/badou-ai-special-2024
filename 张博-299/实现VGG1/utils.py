import matplotlib.image  as mpimg
import numpy as np
import tensorflow as tf
from  tensorflow.python.ops import array_ops
def load_image(path):
    img = mpimg.imread(path)
    short_edge= min(img.shape[:2])
    xx = int((img.shape[1]-short_edge)/2)
    yy = int((img.shape[0] - short_edge) / 2)
    crop_img = img[xx:xx+short_edge,yy:yy+short_edge]
    return crop_img
def resize_image(image,size,
                 align_corners=False,
                 method = tf.image.ResizeMethod.BILINEAR):
    img = tf.expand_dims(image,0)
    img =tf.image.resize_images(image,size,align_corners,method)
    img = tf.reshape(image,tf.stack(-1,size[0],size[1],3))
    return img
def print_prob(prob,file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred =np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print(('Top1:',top1,prob[pred[0]]))
    top5 = [(synset[pred[i],prob[pred[i]]] for i in range(5))]
    print('Top5:',top5)
    return top1





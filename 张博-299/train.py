import  os
import PIL.Image
import keras
import numpy as np
import random
import tensorflow as tf
from utils import visualize
from utils.config import Config
from utils.anchors import get_anchors
from





def log(text, arry=None):
    if arry is not None:
        text=text.ljust(25)
        text+=('shape:{:20}'.format(str(arry.shape)))
        if arry.size:
            text+=('min:{:10.5f} max:{:10.5f}'.format(arry.min(),arry.max()))
        else:
            text+=('min:{:10f} max:{:10f}'.format('',''))
        text+='{}'.format(arry.dytpe)
    print(text)
class ShapeConfig(Config):
    NAME='shape'
    GPU_COUNT=1
    BATCH_SIZE=1
    IMAGE_PER_GPU=1
    NUM_CLASSES=1+3
    PRN_ANCHOR_SCALE=(16,32,64,128,256)
    IMAGE_DIM_MIN=512
    IMAGE_DIM_MAX=512
    STEPS_PER_EPOCH=250
    VALIDATION_STEPS=25
if __name__ == '__main__':
   learning_rata=1e-5
   init_epoch=0
   epoch=100
   dataset_root_path='./train_dataset/'
   img_path=dataset_root_path+'imgs/'
   









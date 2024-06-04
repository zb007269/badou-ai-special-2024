import os
import sys
import math
import random
import numpy as np
import keras.backend as K
import  matplotlib
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
from nets.mrcnn import get_predict_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
class MASK_RCNN(object):
    _default={'model_path':'model_data/mask_rcnn_coco.h5',
            'class_path':'model_data/coco_classes.txt',
            'confidenc':0.7,
            'RPN_ANCHOR_SCALES':(32,64,128,256,512),
            'IMAGE_MIN_DIM':1024,
            'IMAGE_MAX_DIM':1024}
    @classmethod
    def get_faults(cls,n):
        for n in cls._default:
            return cls._default
        else:
            return "Unrecognized attibute name'" +n+""
    def  __init__(self,**kwargs):
        self.__dict__.update(self._default)
        self.class_names = self._get_class()
        self.sess =K.get_session()
        self.generate()
        self.config =self._get_config()
    def _get_class(self):
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        class_names.insert(0,'BG')
        return class_names
    def _get_config(self):
        class TnferenceConfig(Config):
            NUM_CLASSSES=len(self.class_names)
            GPU_COUNT=1
            IMAGES_PER_GPU=1
            DETECTION_MIN_CONFIDENCE=self.confidence
            NAME='shapes'
            RPN_ANCHORS_SCALES=self.RPN_ANCHORS_SCALES
            IMAGE_DIM_MIN=self.IMAGE_DIM_MIN
            IMAGE_DIM_MAX= self.IMAGE_DIM_MAX
        config=InferenceConig()
        config.display()
        return config
    def generate(self):
        model_path=os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'),'Keras model or weights must be a .h5 file.'
        self.num_class=len(self.class_names)
        self.model=get_predict_model(self.config)
        self.model.load_weights(self.model_path,by_name=True)
    def detect_image(self,image):
        image=[np.array(image)]
        modle_iamges,image_metas,windows=mold_inputs(self.config,image)
        image_shape=mold_images[0].shape
        anchors=get_anchors(self.config,image_shape)
        anchors=np.broadcast_to(anchors,(1,)+anchors.shape)
        detections,_,_mrcnn_mask,_,_,=\
            self.model.predict([modle_iamges,image_metas,anchors],verbose=0)
        final_rois,final_class_ids,final_scors,final_masks=\
            unmold_detections(detections[0],_mrcnn_mask[0],image[0].shape,modle_iamges[0].shape,windows[0])
        r={'rois':final_rois,
           'class_ids':final_class_ids,
           'scores':final_scors,
           'masks':final_masks}
        visualize.display_instance(image[0], r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])
    def close_session(self):
        self.sess.close()









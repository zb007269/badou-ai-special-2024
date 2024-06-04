import numpy as np
import math
from  utils.utils import norm_boxes
def generate_config(scales,ratios,shape,feature_stride,anchor_stride):
    scales,ratios=np.meshgrid(np.array(scales),np.array(ratios))
    scales=scales.flatten()
    ratios=ratios.flatten()
    heights=scales/np.sqrt(ratios)
    widths=scales*np.sqrt(ratios)
    shift_x=np.arange(0,shape[1],anchor_stride)*feature_stride
    shift_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shift_y,shift_x =np.meshgrid(shift_y,shift_x)
    box_widths,center_x=np.meshgrid(shift_x,widths)
    box_heights, center_y = np.meshgrid(shift_y, heights)
    boxes_centers=np.stack([center_x,center_y],axis=2).reshape([-1,2])
    boxes_sizes=np.stack([box_widths,box_heights],axis=2).reshape([-1,2])
    boxes=np.concatenate([boxes_centers-0.5*boxes_sizes,boxes_centers+0.5*boxes_sizes],axis=1)
    return boxes
def generate_pyramid_shape(scales,ratios,feature_stride,feature_shapes,anchor_strides):
    anchors=[]
    for i in range(len(scales)):
        anchors.append(scales[i],ratios,feature_stride[i],feature_shapes[i],anchor_strides[i])
    return anchors
def generate_anchor_shape(config,image_shape):
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    assert config.BACKBONE in ['resnet50','resnet101']
    return np.array([[int(math.ceil(image_shape[0]/stride)),int(math.ceil(image_shape[1]/stride))]for stride in config.BACKBONE_STRIDES])
def get_anchor(config,image_shape):
    backbone_shapes=generate_anchor_shape(config,image_shape)
    anchor_cache={}
    if not tuple(image_shape) in anchor_cache:
        a=generate_pyramid_shape(config.RPN_ANCHOR_SCALES,config.BACKBONE_STRIDES,backbone_shapes,config.RPN_AHCHOR_RATIOS,config.BACKBONE_STRIDES)
        anchor_cache[tuple(image_shape)]=norm_boxes(a,image_shape[:2])
    return anchor_cache[tuple(image_shape)]



from keras.layers import Input,ZeroPadding2D,Conv2D,MaxPooling2D,BatchNormalization,Activation,UpSampling2D,Add,Lambda,Concatenate
from keras.layers import Reshape,TimeDistributed,Dense,Conv2DTranspose
from keras.models import Model
import keras.backend as K
from nets.resnet import get_resnet
from nets.layers import ProposalLayer,PyramideRoIAlign,DetectionLayer,DetectionTargetLayer
from net.mrcnn_trainging import *
from utils.anchors import get_anchors
from  utils.utils import nomr_boxes_graph,parse_image_meta_graph
import tensorflow as tf
import numpy as np
def rpn_graph(feature_map,anchors_per_location):
    shared=Conv2D(512,(3,3),padding='same',activation='relu',
                  name='rpn_conv_shared')(feature_map)
    x=Conv2D(2*anchors_per_location,(1,1),padding='valid',
             activation='linear',name='rpn_class_raw')(shared)
    rpn_class_logits=Reshape([-1,2])(x)
    rpn_probs=Activation('softmax',name='rpn_class_xxx')(rpn_class_logits)
    x=Conv2D(anchors_per_location*4,(1,1),padding='valid',
             activation='linear',name='rpn_bbox_pred')(shared)
    rpn_bbox=Reshape([-1,4])(x)
    return [rpn_class_logits,rpn_probs,rpn_bbox]
def build_rpn_model(achors_per_locations,depth):
    input_feature_map=Input(shape=[None,None,depth],name='input_rpn_feature_map')
    outputs=rpn_graph(input_feature_map,achors_per_locations)
    return Model([input_feature_map],outputs,name='rpn_model')
def fpn_classifier_graph(rois,feature_map,image_meta,pool_size,num_classes,train_bn=True,
                         fc_layers_size=1024):
    x=PyramideRoIAlign([pool_size,pool_size],name='roi_align_classifier')([rois,image_meta]+feature_map)
    x=TimeDistributed(Conv2D(fc_layers_size,(pool_size,pool_size),padding='valid'),
                      name='mrcnn_class_conv1')(x)
    x=TimeDistributed(BatchNormalization(),name='mrcnn_class_bn1')(x,training=train_bn)
    x=Activation('relu')(x)
    x=TimeDistributed(Conv2D(fc_layers_size,(1,1)),
                      name='mrcnn_class_conv2')(x)
    x=TimeDistributed(BatchNormalization(),name='mrcnn_class_bn2')(x,training=train_bn)
    x=Activation('relu')(x)
    shared=Lambda(lambda x:K.squeeze(K.squeeze(x,3),2),
                  name='pool_squeeze')(x)
    mrcnn_class_logits=TimeDistributed(Dense(num_classes),
                                       name='mrcnn_class_logits')(shared)
    mrcnn_probs=TimeDistributed(Activation('softmax'),
                                name='mrcnn_class')(mrcnn_class_logits)
    x=TimeDistributed(Dense(num_classes*4,activation='linear'),
                            name='mrcnn_class_logits')(shared)
    mrcnn_bbox=Reshape((-1,num_classes,4),name='mrcnn_bbox')(x)
    return mrcnn_class_logits,mrcnn_probs,mrcnn_bbox
def build_fpn_mask_graph(rois,featur_maps,image_metas,pool_size,num_classes,train_bn=True):
    x = PyramideRoIAlign([pool_size, pool_size], name='roi_align_classifier')([rois,image_metas]+featur_maps)
    x = TimeDistributed(Conv2D(256, (3,3),padding='same',name='mrcnn_class_conv1')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3,3),padding='same',
                        name='mrcnn_class_conv2')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same',
                               name='mrcnn_class_conv3')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn3')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same',
                               name='mrcnn_class_conv4')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn4')(x, training=train_bn)
    x = Activation('relu')(x)
    x=TimeDistributed(Conv2DTranspose(256,(2,2),strides=2,activation='relu'),name='mrcnn_mask_deconv')(x)
    X=TimeDistributed(Conv2D(num_classes,(1,1),strides=1,activation='sigmoid'),name='mrcnn_mask')(x)
    return x
def get_predict_model(config):
    h,w =config.IMAGE_SHAPE[:2]
    if h/2 **6!=int(h/2**6) or w/2**6!=int(w/2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    input_image=Input(shape=[None,None,config.IMAGE_SHAPE[2]],name='input_image')
    input_image_meta=Input(shape=[None,None,config.IMAGE_META_SIZE],name='input_image_meta')
    input_rpn_match=Input(shape=[None,1],name='input_rpn_match',dtype=tf.int32)
    input_rpn_bbox = Input(shape=[None, 4], name='input_rpn_bbox', dtype=tf.int32)
    input_gt_class_ids=Input(shape=[None],name='input_gt_class_ids',dtype=tf.int32)
    input_gt_boxes=Input(shape=[None],name='input_gt_boxes',dtype=tf.int32)
    gt_boxes=Lambda(lambda x:nomr_boxes_graph(x,K.shape(input_image)[1:3]))(input_gt_boxes)
    if config.USE_MINI_MASK:
        input_gt_masks=Input(shape=[config.MINI_MASK_SHAPE[0],config.MINI_MASK_SHAPE[1],None],
                             name='input_gt_masks',dtype=bool)
    else:
        input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPES[1], None],
                               name='input_gt_masks', dtype=bool)
    _,C2,C3,C4,C5=get_resnet(input_image,stage5=True,trian_bn=config.TRAIN_BN)
    P5=Conv2D(config.TOP_DOWN_PYRAMID_SZIE,(1,1),name='fpn_c5p5')(C5)
    P4=Add(name='fpn_p4add')([
        UpSampling2D(size=(2,2),name='fpn_p5upsampled')(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SZIE,(1,1),name='fpn_c4p4')(C4)])
    P3 = Add(name='fpn_p3add')([
        UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SZIE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = Add(name='fpn_p2add')([
        UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SZIE, (1, 1), name='fpn_c2p2')(C2)])
    P2=Conv2D(config.TOP_DOWN_PYRAMID_SZIE,(3,3),padding='same',name='fpn_p2')(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SZIE, (3, 3), padding='same', name='fpn_p3')(P3)
    P4= Conv2D(config.TOP_DOWN_PYRAMID_SZIE, (3, 3), padding='same', name='fpn_p4')(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SZIE, (3, 3), padding='same', name='fpn_p5')(P5)
    P6=MaxPooling2D(pool_size=(1,1),strides=2,name='fpn_p6')(P5)
    rpn_feature_maps=[P2,P3,P4,P5,P6]
    mrcnn_feature_maps=[P2,P3,P4,P5]
    anchors=get_anchors(config.config.IMAGE_SHAPE)
    anchors=np.broadcast_to(anchors,(config.BATCH_SIZE,)+anchors.shape)
    anchors=Lambda(lambda x:tf.Variable(anchors),name='anchors')(input_image)
    rpn=build_rpn_model(len(config.RPN_AHNCOR_RATIOS),config.TOP_DOWN_PYRAMID_SZIE)
    rpn_class_logits,rpn_class,rpn_bbox=[],[],[]
    for p in rpn_feature_maps:
        logits,classes,bbox=rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    rpn_class_logits=Concatenate(axis=1,name='rpn_class_logits')(rpn_class_logits)
    rpn_class=Concatenate(axis=1,name='rpn_class')(rpn_class)
    rpn_bbox=Concatenate(axis=1,name='rpn_bbox')(rpn_bbox)
    proposal_count=config.POST_NMS_ROITS_TRANING
    rpn_rois=ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name='ROI',
        config=config)([rpn_class,rpn_bbox,anchors])
    active_class_ids=Lambda(
        lambda x:parse_image_meta_graph(x)['active_class_ids'])(input_image_meta)
    if not config.USE_RPN_ROIS:
        input_rois=Input(shape=[config.POST_NMS_ROITS_TRANING,4],name='input_roi',dtype=np.int32)
        target_rois=Lambda(lambda x:nomr_boxes_graph(x,K.shape(input_image)[1:3]))(input_rois)
    else:
        target_rois=rpn_rois
    rois,target_class_ids,target_bbox,target_masks=\
        DetectionTargetLayer(config,name='proposal_targets')([
            target_rois,input_gt_class_ids,gt_boxes,input_gt_masks])
    mrcnn_class_logits,mrcnn_class,mrcnn_bbox=\
        fpn_classifier_graph(rois,mrcnn_feature_maps,input_image_meta,
                             config.POOL_SIZE,config.NUM_CLASS,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASS_FC_LAYERS_SIZE)
    mrcnn_mask=build_fpn_mask_graph(rois,mrcnn_feature_maps,input_image_meta,
                                    config.MASK_POOL_SIZE,
                                    config.NUM_CLASS,
                                    train_bn=config.TRAIN_BN)
    output_rois=Lambda(lambda x:x*1,name='output_rois')(rois)
    rpn_class_loss=Lambda(lambda x:rpn_class_loss_graph(*x),name='rpn_class_loss')(
        [input_rpn_match,rpn_class_logits])
    rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config,*x), name='rpn_bbox_loss')(
        [input_rpn_bbox, input_rpn_match,rpn_bbox])
    class_loss=Lambda(lambda x: mrcnn_class_loss_graph(*x), name='mrcnn_class_loss')(
        [target_class_ids, mrcnn_class_logits,active_class_ids])
    bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name='mrcnn_bbox_loss')(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss=Lambda(lambda x: mrcnn_mask_loss_graph(*x), name='mrcnn_mask_loss')(
        [target_masks, target_class_ids, mrcnn_mask])
    inputs=[input_image,input_image_meta,input_rpn_match,
            input_rpn_bbox,input_gt_class_ids,input_gt_boxes,input_gt_masks]
    if not config.USE_RPN_ROIS:
        inputs.append(input_rois)
    outputs=[rpn_class_logits,rpn_class,rpn_bbox,
            mrcnn_class_logits,mrcnn_class,mrcnn_bbox,
            rpn_rois,output_rois,
            rpn_class_loss,rpn_bbox_loss,class_loss,bbox_loss,mask_loss]
    model=Model(inputs,outputs,name='mask_rcnn')
    return  model



















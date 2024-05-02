import warnings
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalMaxPool2D,GlobalAveragePooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
import keras.backend as K
def MobileNet(input_shape=[224,224,3],
              depth_multipier=1,
              dropout=1e-3,
              classes=1000):
    img_input = Input(shape=input_shape)
    x=_conv_block(img_input,32,strides=(2,2))
    x=_depthwise_conv_block(x,64,depth_multipier,block_id=1)
    x=_depthwise_conv_block(x,128,depth_multipier,strides=(2,2),block_id=2)
    x = _depthwise_conv_block(x, 128, depth_multipier, block_id=3)
    x = _depthwise_conv_block(x, 256, depth_multipier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, depth_multipier, block_id=5)
    x = _depthwise_conv_block(x, 512, depth_multipier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, depth_multipier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multipier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multipier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multipier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multipier, block_id=11)
    x = _depthwise_conv_block(x, 1024, depth_multipier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multipier, block_id=13)
    x=GlobalAveragePooling2D()(x)
    x=Reshape((1,1,1024),name='reshape_1')(x)
    x=Dropout(dropout,name='dropout')(x)
    x=Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x=Activation('softmax',name='act_softmax')(x)
    x=Reshape((classes,),name='reshape_2')(x)
    inputs =img_input
    model=Model(inputs,x,name='mobilenet_1_0_224_tf')
    model_name='mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)
    return model
def _conv_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    x =Conv2D(filters,kernel,padding='same',
              use_bias=False,strides=strides,
              name='conv1')(inputs)
    x =BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6,name='conv_relu')(x)
def _depthwise_conv_block(inputs,pointwise_conv_filters,
                          depth_multiplier=1,strides=(1,1),block_id=1):
    x =DepthwiseConv2D((3,3),padding='same',
                       depth_multiplier=depth_multiplier,
                       strides=strides,
                       use_bias=False,
                       name='conv_dw_%d'%block_id)(inputs)
    x =BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
    x=Activation(relu6,name='conv_dw_%d_relu'%block_id)(x)
    x=Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,
             strides=(1,1),name='conv_pw_%d'%block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6,name='conv_pw_%d_relu' % block_id)(x)
def relu6(x):
    return K.relu(x,max_value=6)
def preprocess_input(x):
    x/=225.
    x-=0.5
    x*=2.
    return x
if __name__ =='__main__':
    model=MobileNet(input_shape=(224,224,3))
    img_path='elephant.jpg'
    img =image.load_img(img_path,target_size=(224,224))
    x= image.img_to_array(img)
    x =np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    print('Input image shape:',x.shape)
    preds =model.predict(x)
    print(np.argmax(preds))
    print('Predicted:',decode_predictions(preds,1))


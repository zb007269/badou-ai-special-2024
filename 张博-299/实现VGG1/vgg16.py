import  tensorflow as tf
slim =tf.contrib.slim
def vgg_16(inputs,
           spatial_squeeze=0.5,
           dropout_keep_prob=True,
           is_training=True,
           num_classes=10000,
           scope='vgg_16'):
    with tf.variable_scope('vgg_16',[inputs],scope):
        net = slim.repeat(inputs,2,slim.con2d,64,[3,3],scope='conv1')
        net = slim.max_pool2d(net,[2,2],scope='pool1')
        net = slim.repeat(net, 2, slim.con2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.con2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.con2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.con2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.covn2d(net,4096,[7,7],padding='VALID',scope='fc6')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,
                           scope='dropout6')
        net = slim.covn2d(net, 4096, [1, 1],  scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.covn2d(net,num_classes,[1,1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
            net =tf.squeeze(net,[1,2],name='fc8/squeezed')
        return  net









import tensorflow as tf
import logging
import numpy as np
from . import tf_layer_helper as tlh
from .transform_nets import input_transform_net, feature_transform_net
logger = logging.getLogger(__name__)


def placeholder_inputs(batch_size, num_points, num_features):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud,is_training,config):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    bn_decay = None
    if 'bn_decay' in config['model']:
        bn_decay = config['model']['bn_decay']
    #batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    logger.info(f'point_cloud: {point_cloud.shape}')

    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)

    logger.info(f'transform: {transform.shape}')

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tlh.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tlh.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)

    logger.info(f'transform: {transform.shape}')

    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])

    logger.info(f'point_feat: {point_feat.shape}')

    net = tlh.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tlh.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tlh.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tlh.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    logger.info(f'global_feat: {global_feat.shape}')

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)

    logger.info(f'concat_feat: {concat_feat.shape}')


    net = tlh.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tlh.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tlh.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tlh.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tlh.conv2d(net, len(config['data']['classes']), [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2]) # BxNxC

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.scalar_summary('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.scalar_summary('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
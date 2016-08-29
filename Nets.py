#! usr/bin/env python
"""
Created on Mon May 30 15:58:12 2016
    This file is defining your robot net.
    while you have done, you can set global variable ROBOT_NET=net_filename in file RobotNet.py 
     for training.
@author: shuang
"""

import re
import tensorflow as tf

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
             Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
        
    return loss_averages_op

def conv2d(name, l_input, w, b,stride=[1,1],padding='SAME'):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, stride[0], stride[1], 1], padding=padding),b), name=name)

def max_pool(name, l_input, k ,s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=2):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.0001/5.0, beta=0.75, name=name) 

def RobotNet(images,dropout):
    """
    Build the model for Robot where it will be used as RobotNet.
    Args: 
        images: 4-D tensor with shape [batch_size, height, width, channals].
        dropout: A Python float. The probability that each element is kept.
    Returns:
        Output tensor with the computed classes.
    """
    
#    _X = tf.reshape(images, shape=[-1, IMAGE_HEIGTH, IMAGE_WIDTH, IMAGE_CHANNAL])
#    X = tf.cast(_X, tf.float32)
    X = tf.cast(images, tf.float32)
    
    weights1=tf.Variable(tf.random_normal([11, 11, 3, 96],stddev=0.01))
    biases1=tf.Variable(tf.zeros([96]))
    conv1 = conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')

    norm1 = norm('norm1', conv1, lsize=2)
    
    pool1= max_pool('pool1', norm1, 3, 2)
    
    weights2=tf.Variable(tf.random_normal([5, 5, 96, 256],stddev=0.01))
    biases2=tf.Variable(tf.constant(0.1,shape=[256]))
    conv2 = conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
    
    norm2 = norm('norm2', conv2, lsize=2)
    
    pool2= max_pool('pool2', norm2, 3, 2)
    
    weights3=tf.Variable(tf.random_normal([3, 3, 256, 384],stddev=0.01))
    biases3=tf.Variable(tf.zeros([384]))
    conv3 = conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
    
    weights4=tf.Variable(tf.random_normal([3, 3, 384, 384],stddev=0.01))
    biases4=tf.Variable(tf.constant(0.1,shape=[384]))
    conv4 = conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
    
    weights5=tf.Variable(tf.random_normal([3, 3, 384, 256],stddev=0.01))
    biases5=tf.Variable(tf.constant(0.1,shape=[256]))
    conv5 = conv2d('conv5', conv4, weights5, biases5,stride=[1,1],padding='SAME')
    
    pool5= max_pool('pool5', conv5, 3, 2)
    
    p_h=pool5.get_shape().as_list()[1]
    p_w=pool5.get_shape().as_list()[2]
    print('p_h:',p_h)
    print('p_w:',p_w)
    weights6=tf.Variable(tf.random_normal([p_h*p_w*256, 4096],stddev=0.005))
    biases6=tf.Variable(tf.constant(0.1,shape=[4096]))
    dense1 = tf.reshape(pool5, [-1, weights6.get_shape().as_list()[0]]) 
    fc6= tf.nn.relu(tf.matmul(dense1, weights6) + biases6, name='fc6')

    drop6=tf.nn.dropout(fc6, dropout)
    
    weights7=tf.Variable(tf.random_normal([4096, 4096],stddev=0.005))
    biases7=tf.Variable(tf.constant(0.1,shape=[4096]))
    fc7= tf.nn.relu(tf.matmul(drop6, weights7) + biases7, name='fc7')
    
    drop7=tf.nn.dropout(fc7, dropout)
    
    weights8=tf.Variable(tf.random_normal([4096, 2],stddev=0.01))
    biases8=tf.Variable(tf.zeros([2]))
    net_out= tf.matmul(drop7, weights8) + biases8
    
    saver = tf.train.Saver({v.op.name: v for v in [weights1,biases1,weights2,biases2,weights3,biases3,
                                                   weights4,biases4,weights5,biases5,weights6,biases6,
                                                   weights7,biases7,weights8,biases8]})
    
    return net_out,saver

def RobotNet_simple(images,dropout):
    """
    This is a simple version for RobotNet.
    """
    #_X = tf.reshape(images, shape=[-1, IMAGE_HEIGTH, IMAGE_WIDTH, IMAGE_CHANNAL])
    #X = tf.cast(_X, tf.float32)
    X = tf.cast(images, tf.float32)
    
    weights1=tf.Variable(tf.random_normal([11, 11, 3, 64],stddev=0.01))
    biases1=tf.Variable(tf.zeros([64]))
    conv1 = conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')

    norm1 = norm('norm1', conv1, lsize=2)
    
    pool1= max_pool('pool1', norm1, 3, 2)
    
    weights2=tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=0.01))
    biases2=tf.Variable(tf.constant(0.1,shape=[64]))
    conv2 = conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
    
    norm2 = norm('norm2', conv2, lsize=2)
    
    pool2= max_pool('pool2', norm2, 3, 2)
    
    weights3=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases3=tf.Variable(tf.zeros([64]))
    conv3 = conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
    
    weights4=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases4=tf.Variable(tf.constant(0.1,shape=[64]))
    conv4 = conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
    
    weights5=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases5=tf.Variable(tf.constant(0.1,shape=[64]))
    conv5 = conv2d('conv5', conv4, weights5, biases5,stride=[1,1],padding='SAME')
    
    pool5= max_pool('pool5', conv5, 3, 2)
    
    p_h=pool5.get_shape().as_list()[1]
    p_w=pool5.get_shape().as_list()[2]
    print('p_h:',p_h)
    print('p_w:',p_w)
    weights6=tf.Variable(tf.random_normal([p_h*p_w*64, 1024],stddev=0.005))
    biases6=tf.Variable(tf.constant(0.1,shape=[1024]))
    dense1 = tf.reshape(pool5, [-1, weights6.get_shape().as_list()[0]]) 
    fc6= tf.nn.relu(tf.matmul(dense1, weights6) + biases6, name='fc6')

    drop6=tf.nn.dropout(fc6, dropout)
    
    weights7=tf.Variable(tf.random_normal([1024, 1024],stddev=0.005))
    biases7=tf.Variable(tf.constant(0.1,shape=[1024]))
    fc7= tf.nn.relu(tf.matmul(drop6, weights7) + biases7, name='fc7')
    
    drop7=tf.nn.dropout(fc7, dropout)
    
    weights8=tf.Variable(tf.random_normal([1024, 2],stddev=0.01))
    biases8=tf.Variable(tf.zeros([2]))
    net_out= tf.matmul(drop7, weights8) + biases8
    
    saver = tf.train.Saver({v.op.name: v for v in [weights1,biases1,weights2,biases2,weights3,biases3,
                                                   weights4,biases4,weights5,biases5,weights6,biases6,
                                                   weights7,biases7,weights8,biases8]})
    
    return net_out,saver

def RobotNet_v1(images,dropout):
    """
    decrease a convolution layer.
    """
    X = tf.cast(images, tf.float32)
    
    weights1=tf.Variable(tf.random_normal([11, 11, 3, 64],stddev=0.01))
    biases1=tf.Variable(tf.zeros([64]))
    conv1 = conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')

    norm1 = norm('norm1', conv1, lsize=2)
    
    pool1= max_pool('pool1', norm1, 3, 2)
    
    weights2=tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=0.01))
    biases2=tf.Variable(tf.constant(0.1,shape=[64]))
    conv2 = conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
    
    norm2 = norm('norm2', conv2, lsize=2)
    
    pool2= max_pool('pool2', norm2, 3, 2)
    
    weights3=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases3=tf.Variable(tf.zeros([64]))
    conv3 = conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
    
#    weights4=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
#    biases4=tf.Variable(tf.constant(0.1,shape=[64]))
#    conv4 = conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
    
    weights5=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases5=tf.Variable(tf.constant(0.1,shape=[64]))
    conv5 = conv2d('conv5', conv3, weights5, biases5,stride=[1,1],padding='SAME')
    
    pool5= max_pool('pool5', conv5, 3, 2)
    
    p_h=pool5.get_shape().as_list()[1]
    p_w=pool5.get_shape().as_list()[2]
    print('p_h:',p_h)
    print('p_w:',p_w)
    weights6=tf.Variable(tf.random_normal([p_h*p_w*64, 1024],stddev=0.005))
    biases6=tf.Variable(tf.constant(0.1,shape=[1024]))
    dense1 = tf.reshape(pool5, [-1, weights6.get_shape().as_list()[0]]) 
    fc6= tf.nn.relu(tf.matmul(dense1, weights6) + biases6, name='fc6')

    drop6=tf.nn.dropout(fc6, dropout)
    
    weights7=tf.Variable(tf.random_normal([1024, 1024],stddev=0.005))
    biases7=tf.Variable(tf.constant(0.1,shape=[1024]))
    fc7= tf.nn.relu(tf.matmul(drop6, weights7) + biases7, name='fc7')
    
    drop7=tf.nn.dropout(fc7, dropout)
    
    weights8=tf.Variable(tf.random_normal([1024, 2],stddev=0.01))
    biases8=tf.Variable(tf.zeros([2]))
    net_out= tf.matmul(drop7, weights8) + biases8
    
    saver = tf.train.Saver({v.op.name: v for v in [weights1,biases1,weights2,biases2,weights3,biases3,
                                                   weights5,biases5,weights6,biases6,
                                                   weights7,biases7,weights8,biases8]})
    
    return net_out,saver

def RobotNet_v2(images,dropout):
    """
    decrease a convolution layer, a full connection layer.
    """
    X = tf.cast(images, tf.float32)
    
    weights1=tf.Variable(tf.random_normal([11, 11, 3, 64],stddev=0.01))
    biases1=tf.Variable(tf.zeros([64]))
    conv1 = conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')

    norm1 = norm('norm1', conv1, lsize=2)
    
    pool1= max_pool('pool1', norm1, 3, 2)
    
    weights2=tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=0.01))
    biases2=tf.Variable(tf.constant(0.1,shape=[64]))
    conv2 = conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
    
    norm2 = norm('norm2', conv2, lsize=2)
    
    pool2= max_pool('pool2', norm2, 3, 2)
    
    weights3=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases3=tf.Variable(tf.zeros([64]))
    conv3 = conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
    
#    weights4=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
#    biases4=tf.Variable(tf.constant(0.1,shape=[64]))
#    conv4 = conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
    
    weights5=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases5=tf.Variable(tf.constant(0.1,shape=[64]))
    conv5 = conv2d('conv5', conv3, weights5, biases5,stride=[1,1],padding='SAME')
    
    pool5= max_pool('pool5', conv5, 3, 2)
    
    p_h=pool5.get_shape().as_list()[1]
    p_w=pool5.get_shape().as_list()[2]
    print('p_h:',p_h)
    print('p_w:',p_w)
    weights6=tf.Variable(tf.random_normal([p_h*p_w*64, 1024],stddev=0.005))
    biases6=tf.Variable(tf.constant(0.1,shape=[1024]))
    dense1 = tf.reshape(pool5, [-1, weights6.get_shape().as_list()[0]]) 
    fc6= tf.nn.relu(tf.matmul(dense1, weights6) + biases6, name='fc6')

    drop6=tf.nn.dropout(fc6, dropout)
    
#    weights7=tf.Variable(tf.random_normal([1024, 1024],stddev=0.005))
#    biases7=tf.Variable(tf.constant(0.1,shape=[1024]))
#    fc7= tf.nn.relu(tf.matmul(drop6, weights7) + biases7, name='fc7')
#    
#    drop7=tf.nn.dropout(fc7, dropout)
    
    weights8=tf.Variable(tf.random_normal([1024, 2],stddev=0.01))
    biases8=tf.Variable(tf.zeros([2]))
    net_out= tf.matmul(drop6, weights8) + biases8
    
    saver = tf.train.Saver({v.op.name: v for v in [weights1,biases1,weights2,biases2,weights3,biases3,
                                                   weights5,biases5,weights6,biases6,
                                                   weights8,biases8]})
    
    return net_out,saver

def RobotNet_v3(images,dropout):
    """
    decrease two convolutions layer and a full connection layer.
    """
    X = tf.cast(images, tf.float32)
    
    weights1=tf.Variable(tf.random_normal([11, 11, 3, 64],stddev=0.01))
    biases1=tf.Variable(tf.zeros([64]))
    conv1 = conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')
    _activation_summary(conv1)

    norm1 = norm('norm1', conv1, lsize=2)
    
    pool1= max_pool('pool1', norm1, 3, 2)
    
    weights2=tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=0.01))
    biases2=tf.Variable(tf.constant(0.1,shape=[64]))
    conv2 = conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
    _activation_summary(conv2)
    
    norm2 = norm('norm2', conv2, lsize=2)
    
    pool2= max_pool('pool2', norm2, 3, 2)
    
#    weights3=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
#    biases3=tf.Variable(tf.zeros([64]))
#    conv3 = conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
#    
#    weights4=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
#    biases4=tf.Variable(tf.constant(0.1,shape=[64]))
#    conv4 = conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
    
    weights5=tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.01))
    biases5=tf.Variable(tf.constant(0.1,shape=[64]))
    conv5 = conv2d('conv5', pool2, weights5, biases5,stride=[1,1],padding='SAME')
    _activation_summary(conv5)
    
    pool5= max_pool('pool5', conv5, 3, 2)
    
    p_h=pool5.get_shape().as_list()[1]
    p_w=pool5.get_shape().as_list()[2]
    print('p_h:',p_h)
    print('p_w:',p_w)
    weights6=tf.Variable(tf.random_normal([p_h*p_w*64, 1024],stddev=0.005))
    biases6=tf.Variable(tf.constant(0.1,shape=[1024]))
    dense1 = tf.reshape(pool5, [-1, weights6.get_shape().as_list()[0]]) 
    fc6= tf.nn.relu(tf.matmul(dense1, weights6) + biases6, name='fc6')
    _activation_summary(fc6)

    drop6=tf.nn.dropout(fc6, dropout)
    
#    weights7=tf.Variable(tf.random_normal([1024, 1024],stddev=0.005))
#    biases7=tf.Variable(tf.constant(0.1,shape=[1024]))
#    fc7= tf.nn.relu(tf.matmul(drop6, weights7) + biases7, name='fc7')
#    
#    drop7=tf.nn.dropout(fc7, dropout)
    
    weights8=tf.Variable(tf.random_normal([1024, 2],stddev=0.01))
    biases8=tf.Variable(tf.zeros([2]))
    net_out= tf.matmul(drop6, weights8) + biases8
    _activation_summary(net_out)
    
    saver = tf.train.Saver({v.op.name: v for v in [weights1,biases1,weights2,biases2,
                                                   weights5,biases5,weights6,biases6,weights8,biases8]})
    
    return net_out,saver

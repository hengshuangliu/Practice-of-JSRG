#! usr/bin/env python
"""
Created on Wed May 11 16:56:23 2016
Builds the  RobotNet.
@author: shuang
"""
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import time
#import numpy as np
import os.path
import Robot_data
import Nets

#---------------------------------------Configure---------------------------------------------------------
# parameters for debug model
# if you are not a programmer, please set debug=False.
debug=False
# global parameters for what you want to run.
# ==== 1: for run_trainning
# ==== 2: for run_testing
RUN_FUNCTION=2

# the RobotNet function.
ROBOT_NET=Nets.RobotNet_v3
# Basic parameters.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_string('train_dir', '/home/wangdong/PythonCode/AI/converted_data/0522','Directory with the training data.')
flags.DEFINE_string('model', 'RobotNet_v3','name of model for saver.') 
flags.DEFINE_string('saver_dir', '/home/wangdong/PythonCode/AI/saver/multi_pic/17-4','directory for checkpoint file.') 
flags.DEFINE_string('summary_dir', '/home/wangdong/PythonCode/AI/summary/multi_pic_manul/16_new','directory for summary writer.')

# Basic model parameters for train mode.
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('max_steps', 10000, 'steps for trainnig')   
TRAIN_FILE_LIST=['sponge1.tfrecords','sponge2.tfrecords','sponge3.tfrecords','sponge4.tfrecords',
                'sponge6.tfrecords','sponge7.tfrecords','sponge8.tfrecords',
                'sponge16.tfrecords','sponge18.tfrecords',
                'sponge11.tfrecords','sponge12.tfrecords','sponge13.tfrecords','sponge14.tfrecords',
                'sponge19.tfrecords','sponge10.tfrecords','sponge20.tfrecords']  # .tfrecords files for trainning, string list.
TRAIN_PROB=0.5

# basic parameters for test mode.              
TEST_FILE_LIST=['validation0522.tfrecords']  # .tfrecords files for testing, string list.
flags.DEFINE_integer('test_batch_size', 10, 'Batch size.')
flags.DEFINE_integer('test_numbers', 2000, 'Numbers of testing.')
TEST_PROB=1.0

# The ROBOT images are always 240*320*3 pixels.
IMAGE_HEIGTH=240
IMAGE_WIDTH=320
IMAGE_CHANNAL=3
IMAGE_PIXELS =IMAGE_HEIGTH*IMAGE_WIDTH*IMAGE_CHANNAL
CLASSES=2

# do pre_process for inputed image.
CROP_HEIGTH=240
CROP_WIDTH=240
CROP_CHANNAL=3
IF_RANDOM_CROP=False
IF_FLIP=False
IF_CONTRAST=False
IF_BRIGHT=False
IF_WHITEN=False

#-------------------------------------------------Functions--------------------------------------------------
def loss(net_out, labels):
    """Calculates the loss from the net_out and the labels.
    Args:
        net_out: tensor, float - [batch_size, NUM_CLASSES].
        labels: tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.  
    """
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net_out, labels, name="softmax")
    loss = tf.reduce_mean(cross_entropy, name='reduce_mean')
    return loss

def test_loss():
    with tf.Graph().as_default():
        net_out_i = tf.constant([[1,0],[0.5,0.5],[1,0.5]])
        net_out = tf.cast(net_out_i, tf.float32)
        labels_i = tf.constant([[1,0],[1,0],[0,1]])
        labels = tf.cast(labels_i, tf.float32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net_out, labels, name="softmax")
        loss = tf.reduce_mean(cross_entropy, name='reduce_mean')
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            loss_r, cross_entropy_r = sess.run([loss,cross_entropy])
            print('loss_r:',loss_r)
            print('entre_r:',cross_entropy_r)
    return True
        

def train_op(loss, learning_rate):
    """
    Sets up the training Ops.
    
    Creates a summarizer to track the loss over time in TensorBoard.
    
    Creates an optimizer and applies the gradients to all trainable variables.
    
    The Op returned by this function is what must be passed to the 
    `sess.run()` call to cause the model to train.
    
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(net_out, labels):
    """Evaluate the quality of the net_out at predicting the label.

     Args:
         net_out: net_out tensor, float - [batch_size, NUM_CLASSES].
         labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
    Returns:
         accuracy in a batch with a float32.
    """
    correct_pred = tf.equal(tf.argmax(net_out,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def test_evaluation():
    with tf.Graph().as_default():
        net_out_i = tf.constant([[1,0],[0,1],[1,0]])
        net_out = tf.cast(net_out_i, tf.float32)
        labels_i = tf.constant([[1,0],[1,0],[0,1]])
        labels = tf.cast(labels_i, tf.float32)
        correct_pred = tf.equal(tf.argmax(net_out,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            c_r,accur_r = sess.run([correct_pred,accuracy])
            print('c_r:',c_r)
            print('accur_r:',accur_r)
    return True

def _check_dir(chk_dir):
    """
    check if chk_dir is already existed. if not, create it.
    Args:
        chk_dir: string, directory to be checking.
    """
    if os.path.exists(chk_dir):
        if os.path.isabs(chk_dir):
            print("%s is an absolute path"%(chk_dir))
        else:
            print("%s is a relative path"%(chk_dir))
    else:
        print(chk_dir+" is not existed.")
        os.mkdir(chk_dir)
        print(chk_dir+" is created.")
    return True

def run_training():
    """
    Run the train for RobotNet.
    """
    with tf.Graph().as_default():
        R_data=Robot_data.Robot_data(data_dir=FLAGS.train_dir,filename_list=TRAIN_FILE_LIST,batch_size=FLAGS.batch_size,
                                     imshape=[IMAGE_HEIGTH,IMAGE_WIDTH,IMAGE_CHANNAL],crop_shape=[CROP_HEIGTH,CROP_WIDTH,CROP_CHANNAL],
                                    if_random_crop=IF_RANDOM_CROP,if_flip=IF_FLIP,if_bright=IF_BRIGHT, if_contrast=IF_CONTRAST, 
                                    if_whiten=IF_WHITEN, num_classes=CLASSES,num_epochs=FLAGS.num_epochs)
        n_images,n_labels=R_data.one_hot_input()
        #x_images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        #y_labels = tf.placeholder(tf.float32, [None, CLASSES])
        x_images = n_images
        y_labels = n_labels
        tf.image_summary('images', x_images,max_images=FLAGS.batch_size)
        #keep_prob = tf.placeholder(tf.float32)
        keep_prob = tf.constant(TRAIN_PROB)
        
        #net_out,saver= RobotNet(x_images, keep_prob)
        net_out,saver= ROBOT_NET(x_images, keep_prob)
        loss_out= loss(net_out, y_labels)
        train_op_out = train_op(loss_out, FLAGS.learning_rate)
        eval_correct = evaluation(net_out, y_labels)
        summary_op = tf.merge_all_summaries()
        #saver = tf.train.Saver()
        
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            _check_dir(FLAGS.saver_dir)
            checkpoint = tf.train.get_checkpoint_state(FLAGS.saver_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
                print("train from step one")
            summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, graph=sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                for step in xrange(FLAGS.max_steps):
                    start_time = time.time()
                    #feed_dict={x_images:n_images, y_labels:n_labels, keep_prob:TRAIN_PROB}
                    #_, loss_value = sess.run([train_op_out, loss_out],feed_dict=feed_dict)
                    _, loss_value = sess.run([train_op_out, loss_out])
                    duration = time.time() - start_time
                    if not coord.should_stop():
                        if step % 100 == 0:
                            # Print status to stdout.
                            print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
                            # Update the events file.
                            #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                            summary_str = sess.run(summary_op)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
                            
                        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                            saver.save(sess, FLAGS.saver_dir+'/'+FLAGS.model, global_step=step)
                            # Evaluate against the training set.
                            print('Training Data Eval:')
                            #accuracy = sess.run(eval_correct,feed_dict={x_images:n_images, y_labels:n_labels, keep_prob:TRAIN_PROB})
                            accuracy = sess.run(eval_correct)
                            print("step:%d time:%.3f"%(step,duration))
                            print("accuracy:%.6f"%(accuracy))
                            print("loss:%.3f"%(loss_value))
                            # Evaluate against the test set.
                            print('Test Data Eval:')
                    else:
                        break
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            finally:
                coord.request_stop()
            coord.join(threads)
    print("run_training ok")
    return True

def test_one_train():
    with tf.Graph().as_default():
        R_data=Robot_data.Robot_data(data_dir=FLAGS.train_dir,filename_list=TRAIN_FILE_LIST,batch_size=FLAGS.batch_size,
                                     imshape=[IMAGE_HEIGTH,IMAGE_WIDTH,IMAGE_CHANNAL],crop_shape=[CROP_HEIGTH,CROP_WIDTH,CROP_CHANNAL],
                                    if_random_crop=IF_RANDOM_CROP, if_flip=IF_FLIP,if_bright=IF_BRIGHT, if_contrast=IF_CONTRAST, 
                                    if_whiten=IF_WHITEN, num_classes=CLASSES,num_epochs=FLAGS.num_epochs)
        x_images,y_labels=R_data.one_hot_input()
        keep_prob = tf.constant(TRAIN_PROB)
        
        # fetch all the tensor in the RobotNet for testing.
        # ......by liuhengshuang.
        dropout = keep_prob
        images = x_images
        #_X = tf.reshape(images, shape=[-1, IMAGE_HEIGTH, IMAGE_WIDTH, IMAGE_CHANNAL])
        X = tf.cast(images, tf.float32)
        
        weights1=tf.Variable(tf.random_normal([11, 11, 3, 96],stddev=0.01))
        biases1=tf.Variable(tf.zeros([96]))
        conv1 = Nets.conv2d('conv1', X, weights1, biases1,stride=[4,4],padding='SAME')
    
        norm1 = Nets.norm('norm1', conv1, lsize=2)
        
        pool1= Nets.max_pool('pool1', norm1, 3, 2)
        
        weights2=tf.Variable(tf.random_normal([5, 5, 96, 256],stddev=0.01))
        biases2=tf.Variable(tf.constant(0.1,shape=[256]))
        conv2 = Nets.conv2d('conv2', pool1, weights2, biases2,stride=[1,1],padding='SAME')
        
        norm2 = Nets.norm('norm2', conv2, lsize=2)
        
        pool2= Nets.max_pool('pool2', norm2, 3, 2)
        
        weights3=tf.Variable(tf.random_normal([3, 3, 256, 384],stddev=0.01))
        biases3=tf.Variable(tf.zeros([384]))
        conv3 = Nets.conv2d('conv3', pool2, weights3, biases3,stride=[1,1],padding='SAME')
        
        weights4=tf.Variable(tf.random_normal([3, 3, 384, 384],stddev=0.01))
        biases4=tf.Variable(tf.constant(0.1,shape=[384]))
        conv4 = Nets.conv2d('conv4', conv3, weights4, biases4,stride=[1,1],padding='SAME')
        
        weights5=tf.Variable(tf.random_normal([3, 3, 384, 256],stddev=0.01))
        biases5=tf.Variable(tf.constant(0.1,shape=[256]))
        conv5 = Nets.conv2d('conv5', conv4, weights5, biases5,stride=[1,1],padding='SAME')
        
        pool5= Nets.max_pool('pool5', conv5, 3, 2)
        
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
        
        # all above is code for testing RobotNet variables.
        # if you are not in testing mode, you can comment these and uncomment 
        # line 364: net_out= RobotNet(x_images, keep_prob)
        # ......by liuhengshuang.

        with tf.Session() as sess:
            #net_out= RobotNet(x_images, keep_prob)
            loss_out= loss(net_out, y_labels)
            eval_correct = evaluation(net_out, y_labels)
            init = tf.initialize_all_variables()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for step in xrange(1):
                x_images_r,y_labels_r,net_out_r,loss_out_r,eval_correct_r=sess.run([x_images,y_labels,net_out,loss_out,eval_correct])
                print('x_images_r:',x_images_r)
                print('y_labels_r:',y_labels_r)
                
                # output for testing RobotNet. by liuhengshuang.
                X_r, weights1_r, biases1_r, conv1_r=sess.run([X,weights1,biases1,conv1])
                print('x_r:',X_r)
                #print('weights1_r:',weights1_r)
                #print('biases1_r:',biases1_r)
                print('conv1_r:',conv1_r)
                norm1_r,pool1_r,weights2_r,biases2_r,conv2_r=sess.run([norm1,pool1,weights2,biases2,conv2])
                print('norm1_r:',norm1_r)
                print('pool1_r:',pool1_r)
                #print('weights2_r:',weights2_r)
                #print('biases2_r:',biases2_r)
                print('conv2_r:',conv2_r)
                norm2_r,pool2_r,weights3_r,biases3_r,conv3_r=sess.run([norm2,pool2,weights3,biases3,conv3])
                print('norm2_r:',norm2_r)
                print('pool2_r:',pool2_r)
                #print('weights3_r:',weights3_r)
                #print('biases3_r:',biases3_r)
                print('conv3_r:',conv3_r)
                weights4_r,biases4_r,conv4_r=sess.run([weights4,biases4,conv4])
                #print('weights4_r:',weights4_r)
                #print('biases4_r:',biases4_r)
                print('conv4_r:',conv4_r)
                weights5_r,biases5_r,conv5_r=sess.run([weights5,biases5,conv5])
                #print('weights5_r:',weights5_r)
                #print('biases5_r:',biases5_r)
                print('conv5_r:',conv5_r)
                pool5_r,weights6_r,biases6_r,dense1_r,fc6_r=sess.run([pool5,weights6,biases6,dense1,fc6])
                print('pool5_r:',pool5_r)
                #print('weights6_r:',weights6_r)
                #print('biases6_r:',biases6_r)
                print('dense1_r:',dense1_r)
                print('fc6_r:',fc6_r)
                drop6_r,weights7_r,biases7_r,fc7_r=sess.run([drop6,weights7,biases7,fc7])
                print('drop6_r:',drop6_r)
                #print('weights7_r:',weights7_r)
                #print('biases7_r:',biases7_r)
                print('fc7_r:',fc7_r)
                drop7_r,weights8_r,biases8_r=sess.run([drop7,weights8,biases8])
                print('drop7_r:',drop7_r)
                #print('weights8_r:',weights8_r)
                #print('biases8_r:',biases8_r)
                # output for testing RobotNet. by liuhengshuang.                
                
                print('net_out_r:',net_out_r)
                print('loss_out_r:',loss_out_r)
                print('eval_correct_r:',eval_correct_r)
            coord.request_stop()
            coord.join(threads)
    print("run_training ok")
    return True

def test_RobotNet():
    with tf.Graph().as_default():
        rd=Robot_data.Robot_data(data_dir=FLAGS.train_dir,filename_list=TRAIN_FILE_LIST,batch_size=FLAGS.batch_size,
                                     imshape=[IMAGE_HEIGTH,IMAGE_WIDTH,IMAGE_CHANNAL],crop_shape=[CROP_HEIGTH,CROP_WIDTH,CROP_CHANNAL],
                                    if_random_crop=IF_RANDOM_CROP, if_flip=IF_FLIP,if_bright=IF_BRIGHT, if_contrast=IF_CONTRAST, 
                                    if_whiten=IF_WHITEN, num_classes=CLASSES,num_epochs=FLAGS.num_epochs)
        images, labels =rd.one_hot_input()
        dropout=tf.constant(1.0)
        net_out,_=ROBOT_NET(images,dropout)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images_r,dropout_r,net_out_r,labels_r=sess.run([images,dropout,net_out,labels])
            print("images_r:",images_r,images_r.shape)
            print("dropout_r",dropout_r,dropout_r.shape)
            print("net_out_r:",net_out_r,net_out_r.shape)
            print("labels_r:",labels_r,labels_r.shape)
            coord.request_stop()
            coord.join(threads)
    print("great work")

def run_testing():
    """run testing for trained RobotNet.
    """
    with tf.Graph().as_default():
        R_data=Robot_data.Robot_data(data_dir=FLAGS.train_dir,filename_list=TEST_FILE_LIST,batch_size=FLAGS.test_batch_size,
                                     imshape=[IMAGE_HEIGTH,IMAGE_WIDTH,IMAGE_CHANNAL],crop_shape=[CROP_HEIGTH,CROP_WIDTH,CROP_CHANNAL],
                                    if_random_crop=IF_RANDOM_CROP, if_flip=IF_FLIP,if_bright=IF_BRIGHT, if_contrast=IF_CONTRAST, 
                                    if_whiten=IF_WHITEN,num_classes=CLASSES,num_epochs=FLAGS.num_epochs)
        n_images,n_labels=R_data.one_hot_input()
        
        x_images = n_images
        y_labels = n_labels
        keep_prob = tf.constant(TEST_PROB)
        
        #net_out,saver= RobotNet(x_images, keep_prob)
        net_out,saver= ROBOT_NET(x_images, keep_prob)
        loss_out= loss(net_out, y_labels)
        eval_correct = evaluation(net_out, y_labels)
        
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init) 
            checkpoint = tf.train.get_checkpoint_state(FLAGS.saver_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
                return False
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                total_accuracy=0.0
                for step in xrange(FLAGS.test_numbers):
                    start_time = time.time()
                    if not coord.should_stop():
                        print('-----Testing accuracy----:')
                        accuracy_r,loss_value = sess.run([eval_correct, loss_out])
                        total_accuracy+=accuracy_r
                        duration = time.time() - start_time
                        print("step:%d time:%.3f"%(step,duration))
                        print("accuracy:%.6f"%(accuracy_r))
                        print("loss:%.6f"%(loss_value))
                    else:
                        break
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            finally:
                coord.request_stop()
                print("-----Total accuracy-----:")
                print(total_accuracy/float(step))
            coord.join(threads)
    print('success')
    return True

def main():
    if debug:
        print("debug mode")
        #test_RobotNet()
        #test_loss()
        #test_evaluation()
        test_one_train()
    else:
        if RUN_FUNCTION==1:
            run_training()  
        elif RUN_FUNCTION==2:
            run_testing()
        else:
            print("RUN_FUNCTION set error: 1 for run_training, 2 for run_testing")

if __name__ == '__main__':
    main()

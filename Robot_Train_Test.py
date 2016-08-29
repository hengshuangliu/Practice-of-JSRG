#! usr/bin/env python
"""
Created on 6-24-2016
This python script is developed for Trainning and Testing RobotNet automaticly.
@author: shuang
"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import time
import os
from random import shuffle
#import numpy as np
#import os.path
import Robot_data
import Nets
import logging
import logging.config

#---------------------------------------Configure---------------------------------------------------------
# parameters for debug model
# if you are not a programmer, please set debug=False.
debug=False
# global parameters for what you want to run.
# ==== 1: for robot_train_test().
# ==== 2: for auto_test().
RUN_FUNCTION=2

# the RobotNet function.
ROBOT_NET=Nets.RobotNet_v3
# Basic parameters.
############################################################
# basic parameters for auto_train_test().-----Function 1.
# common parameters for both mode.
F1_DATA_DIR="/home/wangdong/PythonCode/AI/converted_data/multi_pic"
F1_FILE_COM_PREFFIX='sponge'
F1_FILE_NUM_START=1
F1_FILE_NUM_END=20
F1_CROSS_TIMES=5
F1_INTERVAL=2
F1_BEGIN_NUMS=1
F1_TEST_NUMS=3
# parameters for crop image.
F1_CROP_HEIGTH=240
F1_CROP_WIDTH=240
F1_CROP_CHANNAL=3
# parameters for image information.
F1_IMAGE_HEIGTH=240
F1_IMAGE_WIDTH=320
F1_IMAGE_CHANNAL=3
F1_CLASSES=2
# parameters for store relative file.
F1_LOG_DIR="/home/wangdong/PythonCode/AI/log"
F1_LOG_FILENAME='logging.config'
F1_SAVER_DIR="/home/wangdong/PythonCode/AI/saver/multi_pic"
F1_SAVER_NAME="RobotNet_v3"
F1_SUMMARY_DIR="/home/wangdong/PythonCode/AI/summary/multi_pic"
# parameters for train mode.
F1_TRAIN_NUM_EPOCHS=0 # 0 for loop all the time, 1 for loop just once.
F1_TRAIN_LEARN_RATE=0.001
F1_TRAIN_BATCH=8
F1_TRAIN_MAX=10000
F1_TRAIN_IF_RANDOM_CROP=True
F1_TRAIN_IF_FLIP=False
F1_TRAIN_IF_CONTRAST=False
F1_TRAIN_IF_BRIGHT=False
F1_TRAIN_IF_WHITEN=False
F1_TRAIN_PROB=0.5
# parameters for test mode.
F1_TEST_NUM_EPOCHS=1
F1_TEST_BATCH=100
F1_TEST_MAX=20
F1_TEST_IF_RANDOM_CROP=False
F1_TEST_IF_FLIP=False
F1_TEST_IF_CONTRAST=False
F1_TEST_IF_BRIGHT=False
F1_TEST_IF_WHITEN=False
F1_TEST_PROB=1.0

############################################################
# basic parameters for auto_test().-----Function 2.
# basic parameters for logging.
F2_LOG_DIR="/home/wangdong/PythonCode/AI/log"
F2_LOG_FILENAME='logging.config'
# parameters for test mode.
F2_TEST_FILES=['test.tfrecords']
F2_TEST_NUM_EPOCHS=1
F2_TEST_BATCH=10
F2_TEST_MAX=200
F2_TEST_IF_RANDOM_CROP=False
F2_TEST_IF_FLIP=False
F2_TEST_IF_CONTRAST=False
F2_TEST_IF_BRIGHT=False
F2_TEST_IF_WHITEN=False
F2_TEST_PROB=1.0
# common parameters for both mode.
F2_SAVER_DIR="/home/wangdong/PythonCode/AI/saver/multi_pic"
F2_DATA_DIR="/home/wangdong/PythonCode/AI/converted_data/multi_pic"
F2_SAVER_1ST_START=1
F2_SAVER_1ST_END=17
F2_SAVER_1ST_STEP=2
F2_SAVER_2ND_START=0
F2_SAVER_2ND_END=4
F2_SAVER_2ND_STEP=1
# parameters for crop image.
F2_CROP_HEIGTH=240
F2_CROP_WIDTH=240
F2_CROP_CHANNAL=3
# parameters for image information.
F2_IMAGE_HEIGTH=240
F2_IMAGE_WIDTH=320
F2_IMAGE_CHANNAL=3
F2_CLASSES=2
#-------------------------------------------------Functions--------------------------------------------------
# logging configure from logging.config.
logging.config.fileConfig(F1_LOG_FILENAME)
logger = logging.getLogger('shuang')

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
        msg=chk_dir+" is created."
        print(msg)
        logger.info(msg)
    return True

#def _write_to_txt(append_str,dest_dir,dest_file):
#    """write "append_str" to dest_dir/dest_file.
#    """
#    filename = os.path.join(dest_dir,dest_file)
#    if os.path.isfile(filename):
#        with open(filename,'a') as f:
#            print("write to file:",filename)
#            f.write(append_str)
#    else:
#        with open(filename,'w') as f:
#            print("write to file:",filename)
#            f.write(append_str)
#    return True

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

def train(data_dir,filename_list,batch_size,imshape,crop_shape,if_random_crop,
          if_flip,if_bright, if_contrast, if_whiten, num_classes,num_epochs,
          train_prob,learning_rate,summary_dir,saver_dir,max_steps,saver_name):
    """
    Run the train for RobotNet.
    """
    with tf.Graph().as_default():
        R_data=Robot_data.Robot_data(data_dir=data_dir,filename_list=filename_list,batch_size=batch_size,
                                     imshape=imshape,crop_shape=crop_shape,if_random_crop=if_random_crop,
                                     if_flip=if_flip,if_bright=if_bright, if_contrast=if_contrast, 
                                     if_whiten=if_whiten, num_classes=num_classes,num_epochs=num_epochs)
        n_images,n_labels=R_data.one_hot_input()
        #x_images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        #y_labels = tf.placeholder(tf.float32, [None, CLASSES])
        x_images = n_images
        y_labels = n_labels
        tf.image_summary('images', x_images,max_images=batch_size)
        #keep_prob = tf.placeholder(tf.float32)
        keep_prob = tf.constant(train_prob)
        
        #net_out,saver= RobotNet(x_images, keep_prob)
        net_out,saver= ROBOT_NET(x_images, keep_prob)
        loss_out= loss(net_out, y_labels)
        train_op_out = train_op(loss_out,learning_rate)
        eval_correct = evaluation(net_out, y_labels)
        summary_op = tf.merge_all_summaries()
        #saver = tf.train.Saver()
        
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            checkpoint = tf.train.get_checkpoint_state(saver_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
                msg="Successfully loaded:"+ checkpoint.model_checkpoint_path
                logger.info(msg)
            else:
                print("Could not find old network weights")
                logger.info("Could not find old network weights")
                print("train from step one")
            summary_writer = tf.train.SummaryWriter(summary_dir, graph=sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                for step in xrange(max_steps):
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
                            
                        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                            saver.save(sess, saver_dir+'/'+saver_name, global_step=step)
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
                msg='Done training for %d epochs, %d steps.' % (num_epochs, step)
                print(msg)
                logger.info(msg)
            finally:
                coord.request_stop()
            coord.join(threads)
    print("run_training ok")
    return True

def test(data_dir,filename_list,batch_size,imshape,crop_shape,if_random_crop,
         if_flip,if_bright, if_contrast, if_whiten,num_classes,num_epochs,
         test_prob,saver_dir,test_max):
    """run testing for trained RobotNet.
    """
    with tf.Graph().as_default():
        R_data=Robot_data.Robot_data(data_dir=data_dir,filename_list=filename_list,batch_size=batch_size,
                                     imshape=imshape,crop_shape=crop_shape,if_random_crop=if_random_crop, 
                                     if_flip=if_flip,if_bright=if_bright, if_contrast=if_contrast, 
                                    if_whiten=if_whiten,num_classes=num_classes,num_epochs=num_epochs)
        n_images,n_labels=R_data.one_hot_input()
        
        x_images = n_images
        y_labels = n_labels
        keep_prob = tf.constant(test_prob)
        
        #net_out,saver= RobotNet(x_images, keep_prob)
        net_out,saver= ROBOT_NET(x_images, keep_prob)
        loss_out= loss(net_out, y_labels)
        eval_correct = evaluation(net_out, y_labels)
        
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init) 
            checkpoint = tf.train.get_checkpoint_state(saver_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                msg="Successfully loaded:", checkpoint.model_checkpoint_path
                print(msg)
                logger.info(msg)
            else:
                print("Could not find old network weights")
                logger.info(msg)
                return False
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                total_accuracy=0.0
                for step in xrange(test_max):
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
                msg='Done training for %d epochs, %d steps.' % (num_epochs, step)
                print(msg)
                logger.info(msg)
                
            finally:
                coord.request_stop()
                print("-----Total accuracy-----:")
                print(total_accuracy/(step))
                msg="Total accuracy:"+str(total_accuracy/float(step))
                logger.info(msg)
            coord.join(threads)
    print('success')
    return True


def robot_train_test():
    """AUTO run train and test for 20 tf files.
    """
    # check configuretion.
    if (F1_BEGIN_NUMS+F1_TEST_NUMS)>(F1_FILE_NUM_END-F1_FILE_NUM_START+1):
        print("configure error:F1_BEGIN_NUMS+F1_TEST_NUMS>F1_FILE_NUM_END-F1_FILE_NUM_START+1")
        logger.error("configure error:F1_BEGIN_NUMS+F1_TEST_NUMS>F1_FILE_NUM_END-F1_FILE_NUM_START+1")
        return False
    _check_dir(F1_LOG_DIR)
    _check_dir(F1_SAVER_DIR)
    _check_dir(F1_SUMMARY_DIR)
    # file name list.
    file_list=[]
    for num in xrange(F1_FILE_NUM_START,F1_FILE_NUM_END+1):
        filename=F1_FILE_COM_PREFFIX+str(num)+'.tfrecords'
        file_list.append(filename)
        print(file_list)
        msg='all files: '+','.join(file_list)
        logger.info(msg)
        
    file_nums=F1_BEGIN_NUMS 
    while( (file_nums+F1_TEST_NUMS) <= (F1_FILE_NUM_END-F1_FILE_NUM_START+1) ):
        for index in xrange(F1_CROSS_TIMES):
            shuffle(file_list)
            
            filename_list=file_list[0:file_nums]
            summary_dir=F1_SUMMARY_DIR+'/'+str(file_nums)+'-'+str(index)
            saver_dir=F1_SAVER_DIR+'/'+str(file_nums)+'-'+str(index)
            saver_name=F1_SAVER_NAME+str(file_nums)+'-'+str(index)
            _check_dir(summary_dir)
            _check_dir(saver_dir)
            msg="Train start: file:"+str(file_nums)+'-'+str(index)
            logger.info(msg)
            msg='train files:'+','.join(filename_list)
            logger.info(msg)
            if train(data_dir=F1_DATA_DIR,
                     filename_list=filename_list,
                     batch_size=F1_TRAIN_BATCH,
                     imshape=[F1_IMAGE_HEIGTH,F1_IMAGE_WIDTH,F1_IMAGE_CHANNAL],
                     crop_shape=[F1_CROP_HEIGTH,F1_CROP_WIDTH,F1_CROP_CHANNAL],
                     if_random_crop=F1_TRAIN_IF_RANDOM_CROP,
                     if_flip=F1_TRAIN_IF_FLIP,
                     if_bright=F1_TRAIN_IF_BRIGHT,
                     if_contrast=F1_TRAIN_IF_CONTRAST, 
                     if_whiten=F1_TRAIN_IF_WHITEN, 
                     num_classes=F1_CLASSES,
                     num_epochs=F1_TRAIN_NUM_EPOCHS,
                     train_prob=F1_TRAIN_PROB,
                     learning_rate=F1_TRAIN_LEARN_RATE, 
                     summary_dir=summary_dir,
                     saver_dir=saver_dir,
                     max_steps=F1_TRAIN_MAX,
                     saver_name=saver_name):
                         print('train ok')
                         msg="Train success: file:"+str(file_nums)+'-'+str(index)
                         logger.info(msg)
            else:
                print("train failed.")
                logger.info("train failed.")
                return False
            
            msg="Test start: file:"+str(file_nums)+'-'+str(index)
            logger.info(msg)
            test_file_list=file_list[file_nums:(file_nums+F1_TEST_NUMS)]
            msg='test files:'+','.join(test_file_list)
            logger.info(msg)
            if test(data_dir=F1_DATA_DIR,
                     filename_list=test_file_list,
                     batch_size=F1_TEST_BATCH,
                     imshape=[F1_IMAGE_HEIGTH,F1_IMAGE_WIDTH,F1_IMAGE_CHANNAL],
                     crop_shape=[F1_CROP_HEIGTH,F1_CROP_WIDTH,F1_CROP_CHANNAL],
                     if_random_crop=F1_TRAIN_IF_RANDOM_CROP,
                     if_flip=F1_TEST_IF_FLIP,
                     if_bright=F1_TEST_IF_BRIGHT,
                     if_contrast=F1_TEST_IF_CONTRAST, 
                     if_whiten=F1_TEST_IF_WHITEN, 
                     num_classes=F1_CLASSES,
                     num_epochs=F1_TEST_NUM_EPOCHS,
                     test_prob=F1_TEST_PROB,
                     saver_dir=saver_dir,
                     test_max=F1_TEST_MAX):
                         print("test ok")
                         msg="Test end: file:"+str(file_nums)+'-'+str(index)
                         logger.info(msg)
            else:
                print("test failed.")
                logger.error("test failed.")
                return False
        file_nums = file_nums + F1_INTERVAL
    print("this is a great work.")
    logger.info("this is a great work.")
    return True


def auto_test():
    """AUto test for many trained robot_nets.
    """
    msg="All test files:"+','.join(F2_TEST_FILES)
    logger.info(msg)
    for first in xrange(F2_SAVER_1ST_START,F2_SAVER_1ST_END+1,F2_SAVER_1ST_STEP):
        for second in xrange(F2_SAVER_2ND_START,F2_SAVER_2ND_END+1,F2_SAVER_2ND_STEP):
            saver_dir=F2_SAVER_DIR+'/'+str(first)+'-'+str(second)
            msg="Test start for saver:"+str(first)+'-'+str(second)
            logger.info(msg)
            if test(data_dir=F2_DATA_DIR,
                     filename_list=F2_TEST_FILES,
                     batch_size=F2_TEST_BATCH,
                     imshape=[F2_IMAGE_HEIGTH,F2_IMAGE_WIDTH,F2_IMAGE_CHANNAL],
                     crop_shape=[F2_CROP_HEIGTH,F2_CROP_WIDTH,F2_CROP_CHANNAL],
                     if_random_crop=F2_TEST_IF_RANDOM_CROP,
                     if_flip=F2_TEST_IF_FLIP,
                     if_bright=F2_TEST_IF_BRIGHT,
                     if_contrast=F2_TEST_IF_CONTRAST, 
                     if_whiten=F2_TEST_IF_WHITEN, 
                     num_classes=F2_CLASSES,
                     num_epochs=F2_TEST_NUM_EPOCHS,
                     test_prob=F2_TEST_PROB,
                     saver_dir=saver_dir,
                     test_max=F2_TEST_MAX):
                         print("test ok")
                         msg="Test end for files:"+str(first)+'-'+str(second)
                         logger.info(msg)
            else:
                print("test failed.")
                logger.error("test failed.")
                return False
    print("this is a great work.")
    logger.info("this is a great work.")
    return True

def main():
    if debug:
        print("debug mode")
        
    else:
        if RUN_FUNCTION==1:
            robot_train_test()  
        elif RUN_FUNCTION==2:
            auto_test()
        else:
            print("RUN_FUNCTION set error: 1 for robot_train_test(), 2 for auto_test")

if __name__ == '__main__':
    main()

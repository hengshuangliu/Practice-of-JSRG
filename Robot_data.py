#! usr/bin/env python
"""
Created on Tue May 10 17:10:15 2016

Functions for reading data from .tfrecord file.
@author: shuang
"""
from __future__ import absolute_import
from __future__ import print_function
 
import os.path
import tensorflow as tf
import tf_inputs
import utils
#import numpy as np

def convert_one_hot(label_batch, num_labels):
    """convert label to one-hot label.
    Args:
        label_batch: label tensor with shape [batch_size, 1]
        num_labels: numbers of labels,or classes. int scalar.
    Returns:
        labels tensor with shape [batch_size, num_labels]
    """
    sparse_labels = tf.reshape(label_batch, [-1, 1])
    derived_size = tf.shape(label_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_labels])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    return labels

class Robot_data(object):
    """
    Function: read image data from .tfrecords file. and provide for Net trainning.
    raw_label_input() return labels with shape [batch];
    one_hot_input() return one-hot labels with shape [batch,num_labels]
    @author: shuang
    """
    def __init__(self,data_dir,filename_list,batch_size,imshape,num_classes,crop_shape=[0,0,0], if_random_crop=True, 
                 if_flip=False, if_bright=False, if_contrast=False, if_whiten=False, num_epochs=0):
        """
        Args:
            data_dir: string, directory for .tfrecords file.
            filename: string, filename for .tfrecords file.
            batch_size: int, output numbers of images for one time.
            imshape: int_list with lenght 3, [heigth,width,channal]
            crop_shape: if Ture, return image tensor with shape crop_shape.
            if_...: process for image.
            num_classes: numbers of image classes.
            num_epochs: default 0, do circuit for .tfrecords file.
        """
        self.data_dir=data_dir
        self.filename_list=filename_list
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.imshape=imshape
        self.crop_shape=crop_shape
        self.if_random_crop=if_random_crop
        self.if_flip=if_flip
        self.if_bright=if_bright
        self.if_contrast=if_contrast
        self.if_whiten=if_whiten
        self.num_classes=num_classes
        
        
    def raw_label_input(self):
        file_list=[]
        for filename in self.filename_list:
            file_list.append(os.path.join(self.data_dir,filename))
        images, labels = tf_inputs.inputs(file_list,self.batch_size,num_epochs=self.num_epochs,num_threads=1,imshape=self.imshape, 
                                          crop_shape=self.crop_shape, if_random_crop=self.if_random_crop, if_flip=self.if_flip, 
                                          if_bright=self.if_bright, if_contrast=self.if_contrast, if_whiten=self.if_whiten)
        return images, labels
    
    def one_hot_input(self):
        """
        Returns:
            images: images tensor with shape [batch,heigth,width,channal].
            labels: one_hot label with shape [batch,num_classes].
        """
        file_list=[]
        for filename in self.filename_list:
            file_list.append(os.path.join(self.data_dir,filename))
        images, labels = tf_inputs.inputs(file_list,self.batch_size,num_epochs=self.num_epochs,num_threads=1,imshape=self.imshape,
                                          crop_shape=self.crop_shape, if_random_crop=self.if_random_crop, if_flip=self.if_flip, 
                                          if_bright=self.if_bright,if_contrast=self.if_contrast, if_whiten=self.if_whiten)
        num_labels=tf.constant(self.num_classes)
        hot_labels=convert_one_hot(labels, num_labels)
        return images, hot_labels
   
def read_one_image(path='/home/wangdong/PythonCode/AI/one_pic/test.JPG',heigth=240,width=320,crop_size=[0,0,0]):
    """
    Function: Read one image data,and resize it.
        base on tensorflow operations.
    Args:
        path: image path.
        height: new heigth for resized image.
        width: new width for resized image.
    Returns:
        image data with shape [heigth,width,channal].
    """
    _, extension = os.path.splitext(path)
    print(extension)
    reader=tf.WholeFileReader()
    if extension.lower() == '.png':
        key, value = reader.read(tf.train.string_input_producer([path]))
        img = tf.image.decode_png(value)
    if extension.lower() == '.jpg':
        key, value = reader.read(tf.train.string_input_producer([path]))
        img = tf.image.decode_jpeg(value)
    else:
        raise TypeError('Error')
    image=tf.image.resize_images(img,heigth,width)
    if crop_size[0]==0:
        size=tf.constant([heigth,width,3], tf.int32)
    else:
        size=tf.constant(crop_size, tf.int32)
    image2=tf.random_crop(image,size)
    return image2

def test_read_one():
    with tf.Graph().as_default():
        mypath='/home/wangdong/PythonCode/AI/one_pic/test.JPG'
        myheigth=240
        mywidth=320
        mycrop_size=[240,240,3]
        image=read_one_image(path=mypath,heigth=myheigth,width=mywidth,crop_size=mycrop_size)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            image_r=sess.run(image)
            print('image_r:',image_r)
            print(image_r.shape)
            utils.displayArray(image_r)
            coord.request_stop()
            coord.join(threads)
    print('great work')
    return True
        
#def dense_to_one_hot(labels_dense, num_classes):
#    """
#    Convert class labels from scalars to one-hot vectors.
#    """
#    num_labels = labels_dense.shape[0]
#    index_offset = np.arange(num_labels) * num_classes
#    labels_one_hot = np.zeros((num_labels, num_classes))
#    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#    print(labels_one_hot[0])
#    return labels_one_hot   
    
def main():
    """
    if you use utils.displayArray to display image, please open an IpythonConsole and exec on by it.
    """
    with tf.Graph().as_default():
        rd=Robot_data('/home/wangdong/PythonCode/AI/converted_data','test_fish.tfrecords',
                      batch_size=2,imshape=[240,320,3],num_classes=2,crop_shape=[240,240,3],
                      if_flip=False,if_bright=False, if_contrast=True, if_transpose=False)
        #images, labels =rd.raw_label_input()
        images, labels =rd.one_hot_input()
        #img=tf.reshape(images,[-1,240,240,3],name='reshape')
        img=images
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step=0
                while not coord.should_stop():
                    if step > 5:
                        coord.request_stop()
                    else:
                        img_r,labels_r= sess.run([img,labels])
                        print(img_r.shape)
                        print(img_r)
                        utils.displayArray(img_r[0])
                        print("labels_r",labels_r)
                    step +=1
            except tf.errors.OutOfRangeError:
                print('Done training for %d steps.' % (step))
            finally:
                coord.request_stop()
            coord.join(threads)
        print('great work')

if __name__ == '__main__':
    main()
    #test_read_one()

        
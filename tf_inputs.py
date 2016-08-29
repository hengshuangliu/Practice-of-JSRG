#! usr/bin/env python
"""
Created on Sun May 15 13:56:43 2016
Read images and labels from .tfrecords file. base on tensorflow.
@author: shuang
"""
import tensorflow as tf
import utils

def read_and_decode(filename_queue, imshape, normalize=False, flatten=True):
    """Decode data from .tfrecords file.
    Args:
        filename_queue:
        imshape: shape of image [height,width,channal]
        normalize: if normalize true, convert pixel value from [0,255] to [-0.5,0.5]
        flatten: if flatten true, output images with shape [height*width*channal]
    Returns: image and label.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
    features={
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
    })
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    if flatten:
        num_elements = 1
        for i in imshape: num_elements = num_elements * i
        print num_elements
        image = tf.reshape(image, [num_elements])
        image.set_shape(num_elements)
    else:
        image = tf.reshape(image, imshape)
        image.set_shape(imshape)
    if normalize:
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    return image, label

def pre_process(image, h_w_c_size, if_random_crop=True, if_flip=False, if_bright=False, 
                if_contrast=False, if_whiten=False):
    """
    Do preprocessing for image data.
    Atgs:
        image: tensor with shape [height,width,channal].
        h_w_size: integer python list.
        if_random_crop: bool. if True, random crop image. otherwise,  Crop the central [height, width] of the image.
        if_flip: bool, if true, flip image up and down randomly or flip left or right randomly.
        if...
        if_whiten:bool. Subtract off the mean and divide by the variance of the pixels.
    Return:
        image tensor with shape [new_height,new_width,channal].
    """
    size=tf.constant(h_w_c_size,tf.int32)
    if if_random_crop:
        image2=tf.random_crop(image,size)
    else:
        image2=tf.image.resize_image_with_crop_or_pad(image, target_height=h_w_c_size[0], 
                                                      target_width=h_w_c_size[1])   
    if if_flip:
        image3=tf.image.random_flip_up_down(image2)
        image4=tf.image.random_flip_left_right(image3)
    else:
        image4=image2
    if if_bright:
        image5=tf.image.random_brightness(image4,max_delta=63)
    else:
        image5=image4
    if if_contrast:
        image6=tf.image.random_contrast(image5,lower=0.2, upper=1.8)
    else:
        image6=image5
    if if_whiten:
        image7=tf.image.per_image_whitening(image6)
    else:
        image7=image6
    return image7

def inputs(file_list, batch_size, num_epochs, num_threads,
    imshape, crop_shape=[0,0,0], if_random_crop=True, if_flip=False, if_bright=False, 
    if_contrast=False, if_whiten=False, normalize=False, num_examples_per_epoch=128):
    """Reads input tfrecord file num_epochs times. Use it for validation.
    Args:
    file_list: The path to the .tfrecords files to be read, string list.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input ckpt, or 0/None to
    train forever.
    num_threads: Number of reader workers to enqueue
    imshape: The shape of image in the format
    crop_shape: randomly crop image.
    if_...: flip, bright, contrast image.
    num_examples_per_epoch:
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
    in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
    a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs:
        num_epochs = None
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
          file_list, num_epochs=num_epochs, name='string_input_producer')
        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue, imshape, normalize=normalize,flatten=False)
        # Convert from [0, 255] -> [-0.5, 0.5] floats. The normalize param in read_and_decode will do the same job.
        # image = tf.cast(image, tf.float32)
        # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # Ensure that the random shuffling has good mixing properties.
        if crop_shape[0]==0:
            crop_shape=imshape
        image=pre_process(image, h_w_c_size=crop_shape, if_random_crop=if_random_crop, if_flip=if_flip, if_bright=if_bright, 
                          if_contrast=if_contrast, if_whiten=if_whiten)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *min_fraction_of_examples_in_queue)
        images, sparse_labels = tf.train.shuffle_batch(
          [image, label], batch_size=batch_size, num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
           # Ensures a minimum amount of shuffling of examples.
           min_after_dequeue=min_queue_examples, name='batching_shuffling')
    return images, sparse_labels
    
def test_pre_process():
    images,sparse_labels=inputs(file_list=['/home/wangdong/PythonCode/AI/converted_data/test.tfrecords'],
                                 batch_size=1,num_epochs=0,num_threads=1,imshape=[240,320,3])
    images_t=tf.reshape(images,shape=[240,320,3])
    image_pro=pre_process(images_t, h_w_c_size=[240,240,3], if_random_crop=True, if_flip=False, if_bright=False, 
                          if_contrast=False, if_whiten=False)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images_r,labels_r=sess.run([images,sparse_labels])
        print 'images_r',images_r,images_r.shape
        print 'labels_r',labels_r,labels_r.shape
        #utils.displayArray(images_r)
#        images_r,labels_r=sess.run([images,sparse_labels])
#        print 'images_r',images_r,images_r.shape
#        print 'labels_r',labels_r,labels_r.shape
        image_pro_r=sess.run(image_pro)
        print 'image_pro_r',image_pro_r,image_pro_r.shape
        #utils.displayArray(image_pro_r)
        coord.request_stop()
        coord.join(threads)
    print 'great work'
    return True
    
def test_inputs():
    images,sparse_labels=inputs(file_list=['/home/wangdong/PythonCode/AI/converted_data/test_fish.tfrecords'],
                                batch_size=1,num_epochs=0,num_threads=1,imshape=[240,320,3],
                                crop_shape=[240,240,3],if_contrast=True,if_flip=True,if_bright=False)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        images_t=tf.reshape(images,shape=[240,240,3])
        threads = tf.train.start_queue_runners(coord=coord)
        images_r,labels_r=sess.run([images_t,sparse_labels])
        print 'images_r',images_r,images_r.shape
        print 'labels_r',labels_r,labels_r.shape
#        print 'heigth_r',heigth_r,heigth_r.shape
#        print 'width_r',width_r,width_r.shape
#        print 'depth_r',depth_r,depth_r.shape
        images_r,labels_r=sess.run([images_t,sparse_labels])
        print 'images_r',images_r,images_r.shape
        print 'labels_r',labels_r,labels_r.shape
        utils.displayArray(images_r)
        coord.request_stop()
        coord.join(threads)
    return True

if __name__=='__main__':
    test_inputs()
    #test_pre_process()
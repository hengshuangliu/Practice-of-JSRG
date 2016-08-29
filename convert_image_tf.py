#! usr/bin/env python
"""
Created on Wed May 11 08:52:18 2016
  Simple library to read all PNG and JPG/JPEG images in a directory
  with TensorFlow buil-in functions to boost speed.

@author: shuang
"""
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import tensorflow as tf
#from PIL import Image
import numpy as np
#import imageflow
from random import shuffle

#---------------------------------------Configure---------------------------------------------------------
# global parameter for controling what you want to run: 
# ====1 for convert_one_folder: convert one folder image to tf file.
# ====2 for run_convert_image_tf: convert two folders image to tf file.
# =====3 for auto_convert_multiple_groups: convert multiple groups with two folders to tf file.
debug=False
RUN_FUNCTION=2


tf.app.flags.DEFINE_string('converted_dir', 'converted_data/multi_pic','Directory to write the converted result')
FLAGS = tf.app.flags.FLAGS

# basic parameters for convert_one_folder function.
IMAGE_DIR='/home/wangdong/PythonCode/AI/data/multi_pic/pic_yes_test2'
DEST_TF_FILENAME='yes_validation2'
ONE_FOLDER_LABEL=1

# basic parameters for run_convert_image_tf function.
IMAGE_DIR1='/home/wangdong/PythonCode/AI/data/multi_pic/pic_yes_test0716'
IMAGE_DIR2='/home/wangdong/PythonCode/AI/data/multi_pic/pic_no_test'
DEST_FILENAME='test'
IMAGE_LABEL1=1
IMAGE_LABEL2=0

# basis parameters for auto_convert_multiple_groups function.
F3_IMAGE_Y_PARENT_DIR='/home/wangdong/PythonCode/AI/data/multi_pic/pic_yes'
F3_IMAGE_N_PARENT_DIR='/home/wangdong/PythonCode/AI/data/multi_pic/pic_no'
F3_DEST_FILENAME='sponge'
F3_Y_LABEL=1
F3_N_LABEL=0
F3_DIRNAME_PREFFIX='pic_'
F3_NAME_NUM_START=1
F3_NAME_NUM_END=20
#-------------------------------------------------Functions--------------------------------------------------
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

def read_images(path,imshape=[240,320]):
    """
    read images from jpeg or png in your path.
    merge folders randomly in which images have same label. 
    Arg:
    path_list: path list for your image, string
    label_list: labels for each folders. warnning: length should be same with path_list
    Return: images array, [-1, height, width, channel];
            labels array, [-1, label].
    shuang
    """
    images=[]
    png_files=[]
    jpeg_files=[]

    reader = tf.WholeFileReader()

    png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
    jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]')) 
    jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))

    print(jpg_files_path)

    for filename in png_files_path:
        png_files.append(filename)
    for filename in jpeg_files_path:
        jpeg_files.append(filename)
    for filename in jpg_files_path:
        jpeg_files.append(filename)
        
    # Decode if there is a PNG file:
    if len(png_files) > 0:
        png_file_queue = tf.train.string_input_producer(png_files)
        pkey, pvalue = reader.read(png_file_queue)
        temp_p_img = tf.image.decode_png(pvalue)
        p_img = tf.image.resize_images(temp_p_img, imshape[0], imshape[1])

    if len(jpeg_files) > 0:
        jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
        jkey, jvalue = reader.read(jpeg_file_queue)
        temp_j_img = tf.image.decode_jpeg(jvalue)
        j_img = tf.image.resize_images(temp_j_img, imshape[0], imshape[1])

    init=tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print('session start')
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        if len(png_files) > 0:
            for i in range(len(png_files)):
                #fetch image value with shape [240,320,3]
                png = p_img.eval()
                images.append(png)
        
        if len(jpeg_files) > 0:
            for i in range(len(jpeg_files)):
                jpeg = j_img.eval()
                if jpeg.shape!=(240,320,3):
                    print('jpeg:',jpeg.shape)
                    print('this images could not satisfy condition')
                else:
                    images.append(jpeg)
                    print(jpeg)
                    print(jpeg.shape)
        coord.request_stop()
        coord.join(threads)
        print("session ok")
    print('start to convert list to np.array')
    images=np.array(images,np.uint8)
    print('convert list to np.array success')
    print(images.shape)
    return images

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    print('labels shape is ', labels.shape[0])
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %(images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    
    if os.path.exists(FLAGS.converted_dir):
        filename = os.path.join(FLAGS.converted_dir, name + '.tfrecords')
    else:
        os.mkdir(FLAGS.converted_dir)
        print("create a directory:"+FLAGS.converted_dir)
        filename = os.path.join(FLAGS.converted_dir, name + '.tfrecords')
        print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

def convert_image_tf(path_list, label_list, name):
    """convert images in PATH list with LABEL into .tfrecord file.
       use read_images and convert_to function.
    Args:
        path_list: path list for your image, string,length=2
        label_list: labels for each folders. warnning: length should be same with path_list.
    Returns:
        True or False.
    """
    try:
        if len(path_list)==2 and len(label_list)==2:
            images=[]
            labels=[]
            shuffle_list=[]
            images_a= read_images(path_list[0])
            print('first folder read success')
            images_b= read_images(path_list[1])
            print('second folder read success')
            for index in range(images_a.shape[0]):
                shuffle_list.append([label_list[0],images_a[index]])
            for index2 in range(images_b.shape[0]):
                shuffle_list.append([label_list[1],images_b[index2]])
            shuffle(shuffle_list)
            for value in shuffle_list:
                print("label:",value[0])
                #print("image:",value[1])
                labels.append(value[0])
                images.append(value[1])
            print("shuffle images ok")
            images=np.array(images)
            labels=np.array(labels)
            print("labels shape:",labels.shape)
            print("images shape:",images.shape)
            
            convert_to(images, labels, name)
            print("convert ok")
            return True
        else:
            raise ValueError("number of path(%d) and number of label (%d) should be 2"%(len(path_list),len(label_list)))
    except ValueError as e:
        print(e)
        return False

def generate_labels(image_num,label,total_classes):
    """
    Function: Generate one-hot label.
    usage: 
        image_num(int)=5,labels(int)=2,total_classes(int)=5
    return:
        [[01000][01000][01000][...][...][...]]
    """
    try:
        if type(image_num)==int and type(total_classes)==int and type(label)==int:
            if 0<=label<total_classes:
                one_label=np.zeros([total_classes])
                one_label[label]=1
                labels=[]
                for i in range(image_num):
                    labels.append(one_label)
                return np.array(labels)
            else:
                raise ValueError("label should be more than 0 and less than total_classes")
        else:
            raise ValueError("Argurment should be integer")
    except ValueError as e:
        print(e)
        return False
    return True   

def convert_one_folder():
    print("start")
    images=read_images(IMAGE_DIR)
    print("generate_lables")
    #labels=np.zeros([images.shape[0]])
    print(images.shape)   
    labels=np.ones([images.shape[0]])*ONE_FOLDER_LABEL
    convert_to(images,labels,DEST_TF_FILENAME)
    print("Finish converting.")

def run_convert_image_tf():
    """label:0 for no
        label:1 for yes
    """
    print("test convert_image_tf()")
    path_list=[IMAGE_DIR1,IMAGE_DIR2]
    label_list=[IMAGE_LABEL1,IMAGE_LABEL2]
    name=DEST_FILENAME
    if convert_image_tf(path_list,label_list,name):
        print("great work")
        return True
    else:
        print("bad work")
        return False

def auto_convert_multiple_groups():
    """FUNCTION3:convert multiple groups with two classes images into tf file.
    """
    for index in xrange(F3_NAME_NUM_START,F3_NAME_NUM_END+1):
        y_dir=F3_IMAGE_Y_PARENT_DIR+'/'+F3_DIRNAME_PREFFIX+str(index)
        n_dir=F3_IMAGE_N_PARENT_DIR+'/'+F3_DIRNAME_PREFFIX+str(index)
        path_list=[y_dir,n_dir]
        
        label_list=[F3_Y_LABEL, F3_N_LABEL]
        name=F3_DEST_FILENAME+str(index)
        if convert_image_tf(path_list, label_list, name):
            print(y_dir+" converted success.")
            print(n_dir+" converted success.")
        else:
            print("bad work.")
            break
    print("great work.")
    return True


if __name__=='__main__':
    if RUN_FUNCTION==1:
        convert_one_folder()
    elif RUN_FUNCTION==2:
        run_convert_image_tf()
    elif RUN_FUNCTION==3:
        auto_convert_multiple_groups()
    else:
        print('RUN_FUNCTION error: 1 for convert_one_folder, 2 for run_convert_image_tf') 
    #read_images('/home/wangdong/PythonCode/AI')
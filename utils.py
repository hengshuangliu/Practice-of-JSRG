#! usr/bin/env python
"""
Created on Fri May 20 09:09:27 2016

merge images and extract a part of them randomly.

@author: shuang
"""
#import tensorflow as tf
import numpy as np
import os
import random
import shutil

import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import time

#---------------------------------------Configure---------------------------------------------------------
#local parameters
debug=False
# 1 for merge: merge files in the two directory into destination directory.
# 2 for random_extract:
# 3 for displayArray:
RUN_FUNCTION=2

# merge function arguments.
IMAGE_DIR1='old.data/pic0529/pic_y2'
IMAGE_DIR2='old.data/pic0529/pic_y3'
MERGE_DIR='old.data/pic0529/pic_y'

# random_extract parameters.
SOURCE_DIR='data/multi_pic/pic_no/pic'
EXTRACT_DIR='data/multi_pic/pic_no/pic_19'
EXTRACT_NUM=250
SUFFIX_LIST=[]  # if string_list is empty, no file limited. for example, suffix=['.txt']
IF_CP=False  # if IF_CP=True, copy file, otherwise move file.

#--------------------------------------Functions---------------------------------------------------------
def merge(merge_dir=MERGE_DIR, image_dir1=IMAGE_DIR1, image_dir2=IMAGE_DIR2):
    """
    Merge files in the two directory into destination directory.
    Args:
        merge_dir: destination directory,string.
        image_dir1: string.
        image_dir2: string.
    Returns: bool, True for success, and False for fail.
    """
    print 'current work directory:',os.getcwd()
    dir1_filename_list=[]
    dir2_filename_list=[]
    try:
        if not os.path.exists(image_dir1):
            raise ValueError('%s is not exist.'%image_dir1)
        if not os.path.exists(image_dir2):
            raise ValueError('%s is not exist.'%image_dir2)
        dir1_filename_list=os.listdir(image_dir1)
        dir2_filename_list=os.listdir(image_dir2)
        same_filename=[]
        if len(dir1_filename_list)==0:
            raise ValueError('%s is empty.'%image_dir1)
        if len(dir2_filename_list)==0:
            raise ValueError('%s is empty'%image_dir2)
        for filename in dir1_filename_list:
            if filename in dir2_filename_list:
                same_filename.append(filename)
        if not os.path.exists(merge_dir):
            print 'merge_dir:',merge_dir,' is not exist.'
            os.mkdir(merge_dir)
            print 'merge_dir:',merge_dir,'is created.'
        if len(same_filename)>0:
            print 'those file have same name in %s and %s'%(image_dir1,image_dir2)
            print same_filename
            if_rename=raw_input('rename them or give up merge them? (r=rename,g=give up):')
            if if_rename=='r':
                for f in dir1_filename_list:
                    shutil.copy(os.path.join(image_dir1,f),merge_dir)
                    if f in same_filename:
                        os.rename(os.path.join(merge_dir,f), os.path.join(merge_dir,'(1)'+f))
                for f2 in dir2_filename_list:
                    shutil.copy(os.path.join(image_dir2,f2),merge_dir)
                    if f2 in same_filename:
                        os.rename(os.path.join(merge_dir,f2), os.path.join(merge_dir,'(2)'+f2))                    
            elif if_rename=='g':
                for f3 in dir1_filename_list:
                    if f3 not in same_filename:
                       shutil.copy(os.path.join(image_dir1,f3),merge_dir)
                for f4 in dir2_filename_list:
                    if f4 not in same_filename:
                       shutil.copy(os.path.join(image_dir2,f4),merge_dir)
            else:
                raise ValueError('Error input: r=rename,g=give up')
        else:
            for f5 in dir1_filename_list:
                shutil.copy(os.path.join(image_dir1,f5),merge_dir)
            for f6 in dir2_filename_list:
                shutil.copy(os.path.join(image_dir2,f6),merge_dir)
    except ValueError as e:
        print e
        return False
    print 'merge success.'
    return True

def random_extract(src_dir=SOURCE_DIR, dest_dir=EXTRACT_DIR, num=EXTRACT_NUM, 
                   suffix_list=SUFFIX_LIST, if_copy=False):
    """
    randomly extract some files in source directory.
    Args:
        src_dir: source directory,string.
        dest_dir: destnation diretory, if not exist, creat it. string.
        num: numbers of file you want to copy or move,integer.
        suffix_list: suffix for your wanted file, string list.
        if_copy: if True, copy file form src_dir to dst_dir.
    Returns: 
    """
    print 'current work directory:',os.getcwd()
    filename_list=[]
    try:
        if not os.path.exists(src_dir):
            raise ValueError('SOURCE_DIR:%s is not exist.'%SOURCE_DIR)
        else:
            file_list=os.listdir(src_dir)
            if len(suffix_list)==0:
                filename_list=file_list
            else:
                if len(file_list)==0:
                    print 'no file in ',src_dir
                    return False
                else:
                    for filename in file_list:
                        if os.path.splitext(filename)[1] in suffix_list:
                            filename_list.append(filename)
            # random copy or cut files.
            if len(filename_list) <= num:
                raise ValueError('extract numbers error:%d should be less than files in %s'%(num,src_dir))
            else:
                if not os.path.exists(dest_dir):
                    print 'dest_dir:',dest_dir,' is not exist.'
                    os.mkdir(dest_dir)
                    print 'dest_dir:',dest_dir,'is created.'
                random.shuffle(filename_list)
                for i in range(num):
                    if if_copy:
                        shutil.copy(os.path.join(src_dir,filename_list[i]), dest_dir)
                    else:
                        shutil.move(os.path.join(src_dir,filename_list[i]), dest_dir)
    except ValueError as e:
        print e
        return False
    print 'great work'
    return True

def displayArray(a, fmt='jpeg', rng=[0,255]):
    """Display an array as a picture.
    Args:
        a: object with array interface.
    """
    a = (a - rng[0])/float(rng[1] - rng[0])*255
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    time.sleep(5)
    clear_output
    return True

def test_display():
    # Initial Conditions -- some rain drops hit a pond

    N=500
    data = np.zeros([N, N], dtype="float32")
    # Some rain drops hit a pond at random points
    for n in range(40):
        a,b = np.random.randint(0, N, 2)
        data[a,b] = np.random.uniform()
    displayArray(data,rng=[-0.1,0.1])
    print 'great work'
    return True

def main():
    if debug:
        print 'debuging'
        test_display()
    else:
        if RUN_FUNCTION==1:
            merge()
        elif RUN_FUNCTION==2:
            random_extract()
        elif RUN_FUNCTION==3:
            displayArray()
        else:
            print 'RUN_FUNCTION setup error:'

if __name__=='__main__':
    main()
#! usr/bin/env python
"""
Created on Fri Apr 29 18:13:09 2016

@author: shuang
"""
import pygame
import pygame.camera
from PIL import Image  
import os

debug=True

class AIcamera(object):
    """
    Get picture from camera.
    """
    
    def __init__(self,begin=0):
        self.pic_name="pic"
        self.pic_counter=begin
        pygame.camera.init()
        self.cam_list=pygame.camera.list_cameras()
        print self.cam_list
        self.cam=pygame.camera.Camera("/dev/video0",(640,480))

    def start(self):
        self.cam.start()
    
    def stop(self):
        self.cam.stop()
    
    def updateCounter(self,counter):
        self.pic_counter=counter
        return True
    
    def savePic(self):
        img=self.cam.get_image()
        if not os.path.exists('./pic'):
            os.mkdir('./pic')
            print 'create directory: pic'
        pygame.image.save(img,'./pic/'+self.pic_name+str(self.pic_counter)+'.jpg')
        self.pic_counter+=1

def compressImage(dstPath,filename):
    """
    Function: compress picture.
    """ 
    if not os.path.exists(dstPath):
            os.makedirs(dstPath)        

    srcFile=os.path.join(dstPath,filename)
    print srcFile

    if os.path.isfile(srcFile):     
        sImg=Image.open(srcFile)  
        w,h=sImg.size  
        print w,h
        dImg=sImg.resize((w/2,h/2),Image.ANTIALIAS)  
        dImg.save(srcFile) 
        print srcFile+" compressed succeeded"
    if os.path.isdir(srcFile):
        compressImage(srcFile,srcFile)


def test():
    if debug:
        Ac=AIcamera()
        Ac.start()
        Ac.savePic()
        Ac.stop()
        print "great work"
        
if __name__=='__main__':
    test()
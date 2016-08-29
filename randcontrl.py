#! usr/bin/env python
"""
Created on Wed May  4 14:54:32 2016

@author: shuang
"""

import instructions
import camera
import random
import time


debug=True
# parameters.
START=101
END=104
INIT_DEGREE=instructions.RESET_INS

class RandmCtl(object):
    """
    Function: random instructions and take a screenshot.
    """
    def __init__(self,minDegree=[20,20,20,20,20,20],maxDegree=[150,130,150,150,150,50],addr='127.0.0.1',port=8001,maxlen=instructions.MAX_LEN):
        self.minDegree=minDegree
        self.maxDegree=maxDegree
        self.maxlen=maxlen
        self.instruction=instructions.Instructions(addr,port)
        self.camera=camera.AIcamera()
        self.degree=[]
        self.counter=0
        self.maxCounter=10

    def init_degree(self):
        for d in INIT_DEGREE:
            self.degree.append(d)
        return True
        
    def start(self):
        self.instruction.connectAI()
        self.camera.start()
        self.init_degree()
        self.degree[-1]=random.randint(self.minDegree[-1],self.maxDegree[-1])
        self.instruction.send(self.degree)
        while True:
            if self.counter<self.maxCounter:
                try:
                    for i in range(self.maxlen):
                        self.degree[i]=random.randint(self.minDegree[i],self.maxDegree[i])
                    self.instruction.send(self.degree)
                    print 'send:',self.degree
                except ValueError as e:
                    print e
                    break
                time.sleep(2)
                self.camera.savePic()
                print 'take a screenshot'
                time.sleep(1)
                self.counter+=1
            else:
                break
            
    def updateCounter(self,counter):
        try:
            if type(counter)==int:
                self.counter=counter
                self.camera.updateCounter(counter)
            else:
                raise ValueError('counter should be integer more than zero')
        except ValueError as e:
            print e
            return False
        return True
    
    def updateMaxCounter(self,maxcounter):
        try:
            if type(maxcounter)==int:
                self.maxCounter=maxcounter
            else:
                raise ValueError('maxcounter should be integer more than zero')
        except ValueError as e:
            print e
            return False
        return True
    
    def stop(self):
        self.camera.stop()
        self.instruction.close()

def test():
    """
    For robot images.
    """
    if debug:
        rdm=RandmCtl([10,50,50,40,120,85],[120,120,140,140,120,85],'192.168.2.3',1024)
        rdm.updateCounter(START)
        rdm.updateMaxCounter(END)
        rdm.start()
        rdm.stop()
        print 'test ok'

if __name__=='__main__':
    test()
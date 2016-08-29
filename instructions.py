#! usr/bin/env python
"""
Created on Wed May  4 10:36:04 2016

@author: shuang
"""

import struct
import binascii
import ctypes
import time
import socketLink

debug=True
# basic parameters: m1=[10,120],m2=[50,120],m3=[0,175],m4=[0,175],m5=[0,175].m6=[30,90]
MAX_LEN=6
RESET_INS=[90,130,160,140,80,60]
LAST_INS=[90,130,160,140,80,60]

class PackIns(object):
    """
    Function:transfer 6 degrees to 32bites instruction data packet.
    One Instruction have 4*6=24 bytes
    every 4 bytes have 32 bites: 0x002_(select:0~5) _ _ _ _(shift degree)
    
    """
    def __init__(self,ins,maxlen):
        self.ins=ins
        self.maxlen=maxlen
        self.intData=[self.degreeToInt(90)]*maxlen
        self.insToIntData()
        self.dataPacket=['0000']*maxlen
        
    def degreeToInt(self,degree=90):
        d=65536*(0.1*degree/180+0.025)
        return int(d)

    def updateIns(self,ins):
        self.ins=ins
        return True
    
    def updateIntData(self, slt, step=36,ifAdd=True):
        try:
            if 0<=slt<6:
                if ifAdd:
                    self.intData[slt]+=step
                else:
                    self.intData[slt]-=step
            else:
                raise ValueError('slt should be integer from 0 to 5')
        except ValueError as e:
            print e
            return False
        return True
        
    def updateIntData_zero(self,intD=[0,0,0,0,0,0]):
        try:
            if len(intD)==MAX_LEN:
                self.intData=intD
            else:
                raise ValueError('length of input list should be 6')
        except ValueError as e:
            print e
            return False
        return True
    
    def insToIntData(self):
        for i in range(self.maxlen):
            self.intData[i]=self.degreeToInt(self.ins[i])
        return True
        
    def pack(self):
        s=struct.Struct('!2H')
        prebuffer=ctypes.create_string_buffer(s.size)
        print 'Before :',binascii.hexlify(prebuffer)
        for i in range(self.maxlen):
            contrlBite=int('0020',16)+i
            value=(contrlBite,self.intData[i])
            s.pack_into(prebuffer,0,*value)
            print 'After pack:',binascii.hexlify(prebuffer)
            self.dataPacket[i]=prebuffer[0]+prebuffer[1]+prebuffer[2]+prebuffer[3]
        return True
    
    def getIns(self):
        """
        return ins, not a new object.
        """
        return self.ins

    def getIntData(self):
        return self.intData
    
    def getDataPacket(self):
        Sdata=''
        for j in self.dataPacket:
            Sdata=Sdata+j
        return Sdata


class Instructions(object):
    """
    Function:(degree1 to degree6) control AI
    """
    def __init__(self,addr='127.0.0.1',port=8001):
        self.addr=addr
        self.port=port
        self.link=socketLink.SocketLink(self.addr,self.port)
        self.resetIns=RESET_INS
        self.lastIns=LAST_INS
    
    def getLastIns(self):
        """
        return a new list equal to LastIns.
        """
        value=[0,0,0,0,0,0]
        for i in range(6):
            value[i]=self.lastIns[i]
        return value
    
    def updateRstIns(self,ins):
        self.resetIns=ins
        return True
    
    def connectAI(self):
        self.link.connect()
    
    def send(self,ins=[90,130,160,140,80,90],step=36):
        packet=PackIns(self.lastIns,MAX_LEN)
        for i in range(len(ins)):
            ifAdd=True if ins[i]>self.lastIns[i] else False
            while True:
                if ifAdd and packet.getIntData()[i]<packet.degreeToInt(ins[i]):
                    packet.updateIntData(i,step,ifAdd)
                    ifExit=False
                elif ifAdd==False and packet.getIntData()[i]>packet.degreeToInt(ins[i]):
                    packet.updateIntData(i,step,ifAdd)
                    ifExit=False
                else:
                    self.lastIns[i]=ins[i]
                    packet.updateIns(self.lastIns)
                    packet.insToIntData()
                    ifExit=True
                packet.pack()
                Sdata=packet.getDataPacket()
                self.link.sendInstru(Sdata)
                if ifExit:
                    break
                time.sleep(0.1)
        print 'last:',self.lastIns
        print 'reset:',self.resetIns
        return True
    
    def reset(self):
        self.send(self.resetIns)
        return True

#    def halt(self): #no use
#        packet=PackIns(self.lastIns,6)
#        packet.updateIntData_zero()
#        packet.pack()
#        Sdata=packet.getDataPacket()
#        self.link.sendInstru(Sdata)
#        return True
         
    def close(self):
        self.reset()
#        self.halt()
        self.link.close()
        print 'close socket'
        return True
    
def test():
    if debug:
        ins=Instructions('192.168.2.3',1024)
        #ins=Instructions()
        ins.connectAI()
        ins.send([90,90,170,170,80,90])
        ins.close()
        print 'test ok'

def test2():
    if debug:
        #ins=Instructions()
        ins=Instructions('192.168.2.3',1024)
        ins.connectAI()
        degree=ins.getLastIns()
        ins.send(degree)
        while True:
            print degree
            slt=input('select machine(1-6):')
            data=input('control degree(0-175):')
            try:
                if type(slt)==int and 0<slt<7 and 0<data<175:
                    degree[slt-1]=int(data)
                    ins.send(degree)
                else:
                    raise ValueError('slt should be integer from 1-6,and data should be integer from 0-175')
            except ValueError as e:
                print e
                break
        ins.close()
        print 'test ok'
                    

if __name__=='__main__':
    #test()
    test2()

        
    
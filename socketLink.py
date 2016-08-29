#! usr/bin/env python
"""
Created on Fri Apr 29 18:13:09 2016

@author: shuang
"""
import socket

debug=True

class SocketLink(object):
    """
    this class is used to link to FPGA by TCP/IP.
    """
    def __init__(self,addr="127.0.0.1",port=8001):
        self.addr=addr
        self.port=port
        self.s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    def connect(self):
        self.s.connect((self.addr,self.port))
        print 'connect ok'
    
    def recv(self):
        buf=self.s.recv(1024)
        print 'recv:',buf
    
    def sendInstru(self,instruction='123412341234123412341234'):
        self.s.send(instruction)
    
    def close(self):
        self.s.close()
    
def test():
    if debug:
        sl=SocketLink("127.0.0.1")
        sl.connect()
        sl.sendInstru()
        #sl.recv()
        sl.close()
        print "great work"
        
if __name__=='__main__':
    test()
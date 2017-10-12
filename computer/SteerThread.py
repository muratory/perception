import threading
import urllib
import Queue
import struct
import socket
from commonDeepDriveDefine import *
from commonDeepDriveTools import *

class SteerThread(commonThreadSocket):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(SteerThread, self).__init__()

        self.steerCommand = 'None'
        self.speed = INITIAL_CAR_SPEED
        self.connected = False


        
    def _handle_STOP(self, cmd):
        print "stop ",self.name
        if self.connected == True:
            self.socket.sendall('stop' + '>')
        
    def _handle_SEND(self, cmd):
        if self.connected == True:
            if cmd.data[0] is 'SPEED':
                self.speed = cmd.data[1]
                if self.speed > 100:
                    self.speed = 100
                elif self.speed < 0:
                    self.speed = 0
                self.socket.sendall('speed' + str(self.speed) + '>')
                
            elif cmd.data[0] is 'STEER_COMMAND':
                self.socket.sendall(cmd.data[1] + '>')
                self.steerCommand = cmd.data[1]
                
            elif cmd.data[0] is 'STEER_ANGLE':
        	    command = 'turn=' + str(cmd.data[1]) + '>'
                    self.socket.sendall(command)
                
            else:
                print 'Steer Command unknown' + str()



import threading
import urllib
import Queue
import struct
from commonDeepDriveDefine import *
from commonDeepDriveTools import *


class VideoThread(commonThreadSocket):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(VideoThread, self).__init__()
        self.stream = None
        self.rcvBytes = ''
        self.lastImage= ''
                
    def _handle_CONNECT(self, cmd):
            # try connection ntil it succeed or close sent
            while True:
                #check if new command comes in
                try:
                    newCmd = self.cmd_q.get(False)
                    if newCmd.type == ClientCommand.STOP:
                        print 'stop ',self.name
                        return
                except Queue.Empty:
                    #we should always be there
                    pass
                    
                try:
                    self.stream = urllib.urlopen(cmd.data)
                    self.reply_q.put(self._success_reply())
                    self.connected = True
                    return
                except IOError as e:
                    pass
                    #print 'Video Connect Error : ' + str(e)
                    #then retry again
                    

    # used to receive jpg image in client thread 
    def _handle_RECEIVE(self, cmd):
        if self.connected == True:
            while True:
                #check first if new command to stop comes in
                try:
                    newCmd = self.cmd_q.get(False)
                    if newCmd.type == ClientCommand.STOP:
                        print 'stop ' + self.name
                        return
                except Queue.Empty:
                    #we should always be there
                    pass
                
                try:
                    #loop until image found or problem
                    self.rcvBytes += self.stream.read(1024)
                    #print 'rcv = ' + str(len(self.rcvBytes))
                    # search for jpg image 
                    a = self.rcvBytes.find('\xff\xd8')
                    b = self.rcvBytes.find('\xff\xd9')
                    if a!=-1 and b!=-1:
                        #image found , send it in receive queue
                        self.lastImage = self.rcvBytes[a:b+2]
                        self.reply_q.put(self._success_reply())
                        #now shift rcvbyte to manage next image
                        self.rcvBytes=self.rcvBytes[b+2:]
    
                except IOError as e:
                    self.reply_q.put(self._error_reply(str(e)))

    def _handle_CLOSE(self, cmd):
        if self.connected == True:
            self.stream.close()
            self.connected = False
        

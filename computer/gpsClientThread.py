import threading
import urllib
import Queue
import struct
import socket
from commonDeepDriveDefine import *
from commonDeepDriveTools import *

class gpsClientThread(commonThreadSocket):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(gpsClientThread, self).__init__()
      

    def _handle_RECEIVE(self, cmd):
        if self.connected == True:
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
                    header_data = self._recv_n_bytes(4)
                    if len(header_data) == 4:
                        msg_len = struct.unpack('<L', header_data)[0]
                        data = self._recv_n_bytes(msg_len)
                        if len(data) == msg_len:
                            self.reply_q.put(self._success_reply(data))
                    else:
                        #for whatever reason the len does not match
                        self.reply_q.put(self._error_reply('gpsClient socket misalignement'))
                        
                except IOError as e:
                    if type(e) == socket.timeout:
                        #this could be possible since we don't want to block the task forever
                        #but could be a real issue to get data in
                        print 'Warning : no data received for gps client'
                        continue
                    else:
                        self.reply_q.put(self._error_reply(str(e)))




from __future__ import division
import socket
import struct
import threading
import Queue
import socket
import numpy as np
import cv2
from math import tan,sqrt


from commonDeepDriveDefine import *

#Max connection to this server
MAXCONN=5



class commonThread(threading.Thread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(commonThread, self).__init__()
        #create a queue for each object 
        self.cmd_q = Queue.Queue()
        self.reply_q = Queue.Queue()
        self.alive = threading.Event()
        self.alive.set()
        
        self.handlers = {
            ClientCommand.CONNECT: self._handle_CONNECT,
            ClientCommand.CLOSE: self._handle_CLOSE,
            ClientCommand.RECEIVE: self._handle_RECEIVE,
            ClientCommand.SEND: self._handle_SEND,
            ClientCommand.STOP: self._handle_STOP,
        }
    
    def run(self):
        while self.alive.isSet():
            try:
                # block for all command and wait on it
                cmd = self.cmd_q.get(True,1)
                #print threading.currentThread().getName() + 'CMD = ' + str(cmd.type)
                self.handlers[cmd.type](cmd)
            except Queue.Empty:
                #no process of CMD queue.empty because it is a regular case
                pass
                
    def join(self, timeout=None):
        #print 'WARNING : queue stopped : ' + threading.currentThread().getName()
        self.alive.clear()
        threading.Thread.join(self, timeout)

    def _handle_CLOSE(self, cmd):
        pass
        
    def _handle_STOP(self, cmd):
        print "stop ",self.name
        pass

    def _handle_CONNECT(self, cmd):
        pass

    def _handle_RECEIVE(self, cmd):
        pass

    def _handle_SEND(self, cmd):
        pass

    def _error_reply(self, errstr):
        return ClientReply(ClientReply.ERROR, errstr)

    def _success_reply(self, data=None):
        return ClientReply(ClientReply.SUCCESS, data)



class commonThreadSocket(commonThread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(commonThreadSocket, self).__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #2 seconde Timeout to receive data
        self.socket.settimeout(2)
        self.connected = False

    def _handle_CLOSE(self, cmd):
        if self.connected == True:
            self.socket.close()
            self.connected = False
            print 'socket closed for ',self.name 
        
    def _handle_STOP(self, cmd):
        print "stop ",self.name
        pass
           
    def _handle_CONNECT(self, cmd):
        # try connection ntil it succeed or close sent
        while True:
            #check if new command comes in
            try:  
                newCmd = self.cmd_q.get(False)
                if newCmd.type == ClientCommand.STOP:
                    return
            except Queue.Empty:
                #we should always be there
                pass
            try:
                self.socket.connect((cmd.data[0], cmd.data[1]))
                self.connected = True
                self.reply_q.put(self._success_reply())
                return
            
            except IOError as e:
                pass
                #print 'Steer Connect Error : ' + str(e)
                #retry again


    def _recv_n_bytes(self, n):
        """ Convenience method for receiving exactly n bytes from self.socket
            (assuming it's open and connected).
        """
        data = ''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if chunk == '':
                break
            data += chunk
        return data        
        
    def _error_reply(self, errstr):
        return ClientReply(ClientReply.ERROR, errstr)

    def _success_reply(self, data=None):
        return ClientReply(ClientReply.SUCCESS, data)



####### class dedicated to all thread/socket control

class ClientCommand(object):
    """ A command to the client thread.
        Each command type has its associated data:

        CONNECT:    (host, port) tuple
        SEND:       Data string
        RECEIVE:    None
        CLOSE:      None
    """
    CONNECT, SEND, RECEIVE, STOP, CLOSE  = range(5)

    def __init__(self, type, data=None):
        self.type = type
        self.data = data


class ClientReply(object):
    """ A reply from the client thread.
        Each reply type has its associated data:

        ERROR:      The error string
        SUCCESS:    Depends on the command - for RECEIVE it's the received
                    data string, for others None.
    """
    ERROR, SUCCESS = range(2)

    def __init__(self, type, data=None):
        self.type = type
        self.data = data


"""
@function      : showLabel(label)
@description   : Display on the picture label stored (value and line)
@param label   : Wheel angle 
@type  label   : int
@param im name : image name used to display label 
@type im_name  : string
@param image   : image used to display label 
@type image    : Array of shape
@return        : None
"""   
def showLabel(label, image_name, image):
    # Draw line with current direction of the wheel
    # x_top_wheel = (IMAGE_PIXELS_X/2) + (float((IMAGE_PIXELS_Y - 70) * tan( label / 0.6 /180.0 * np.pi)))
    x_top_wheel = (int(IMAGE_PIXELS_X/2)) + (float((IMAGE_PIXELS_Y - 70) * tan( label /180.0 * np.pi)))
    cv2.line(image, (int(IMAGE_PIXELS_X/2), IMAGE_PIXELS_Y), (int(x_top_wheel), 70), (0, 255, 0),2)

        
def distance(p1,p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

          

# Class tcpclientThread for client handling
class TcpClientThread(threading.Thread):
        
    def __init__(self, ip, port, conn):
        super(TcpClientThread, self).__init__()
        self.ip = ip
        self.port = port
        self.conn = conn
        self.clientTcpQueue = Queue.Queue()
        self.connected = True
        print 'New Client starting up on %s:%s' % (str(self.ip), str(self.port))

    def run(self):
        while True:
            try:
                # block on queue to get data
                data = self.clientTcpQueue.get(True,0.5)
                if data[0] == 'DATA':
                    #print 'Server received : ' + data[1] + ' to be send to client ' + str(self.ip) + ':' + str(self.port)
                    #send len + payload to the client connected
                    try:
                        self.conn.sendall(struct.pack('<L', len(data[1])))
                        self.conn.sendall(data[1])
                    except IOError:
                        print 'Client Probably disconnected on port ' + str(self.port)
                        self.conn.close()
                        self.connected= False
                        return
                    
                elif data[0] == 'CLOSE':
                    print 'Client disconnection requested on port : ' + str(self.port)
                    self.conn.close()
                    self.connected = False
                    return
                else: 
                    print 'ERROR : Strange data received in queue for client ',  str(self.port)
                    print str(data)
                    self.conn.close()
                    self.connected = False
                    return
                    
            except Queue.Empty:
                #no process of CMD queue.empty because it is a regular case
                #print 'nothing in queue fort client port : ' + str(self.port) 
                pass


# Class tcpserver for tcp connection handling
class TcpServer(threading.Thread):
    def __init__(self,port):
        super(TcpServer, self).__init__()
        self.tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        self.tcpServer.bind(('',port))
        self.tcpServer.settimeout(5.0)
        self.tcpServer.listen(MAXCONN)
        #list of client connected to the server
        self.clientThreads = []
        self.tcpServerActive = True
        self.port = str(port)
        print 'server starting up on port',self.port

    def run(self):
        print 'server ',self.name,' listening on port : ',self.port
        while self.tcpServerActive == True:
            try:

                (conn, (ip, port)) = self.tcpServer.accept()  
                # ADD Client
                newclient = TcpClientThread(ip, port, conn)
                newclient.start()
                self.clientThreads.append(newclient)

                      
            except Exception:
                if len(self.clientThreads) == 0:
                    print 'wait for new client to be connected on port ',self.port
                                
                #do clean up of connection if needed
                for clientTcp in self.clientThreads:
                    if clientTcp.connected == False:
                        print 'remove disconnected client on port ',clientTcp.port
                        self.clientThreads.remove(clientTcp)               
                continue
        
        self.tcpServer.close()
        print 'server Ended'
        
        
    def sendDataToClient(self,data):
        for clientTcp in self.clientThreads:
            #print 'send(' + str(data) + ')to client port ' + str(clientTcp.port)
            clientTcp.clientTcpQueue.put(('DATA',str(data)))

    def closeClientAndServer(self):
        for clientTcp in self.clientThreads:
            print 'close client port ' + str(clientTcp.port)
            clientTcp.clientTcpQueue.put(('CLOSE',0))
        self.tcpServerActive = False



#server thread class used to send data to several client
class serverThread(commonThread):
    def __init__(self):
        super(serverThread, self).__init__()
        self.connected = False


    def _handle_STOP(self, cmd):
        print "stop ",self.name
        if self.connected == True:
            self.serverThread.closeClientAndServer()
            self.connected = False
           
    def _handle_CONNECT(self, cmd):
        try:
            #start tcp server thread
            self.serverThread = TcpServer(cmd.data)
            self.serverThread.start()
            self.reply_q.put(self._success_reply())
            self.connected = True
            return
        
        except IOError as e:
            pass

    def _handle_SEND(self, cmd):
        if self.connected == True:
            self.serverThread.sendDataToClient(str(cmd.data))
                            
"""
@function      : NNroadLabel2Num(imgArray,NNroadLabel = None)
@description   : reshape the image from a 3 quarter to a 2 quarter 120x320 image
                 according to the use case 
@param fileName: array of image or image alone (first shape to discriminate)
               : NNroadLabel : use case road to be use for this image resizing
@rtype         : None
@return        : None
"""         
def NNroadLabel2Num(NNroadLabel):
    if (NNroadLabel == 'IDLE'):
        return 0
    elif (NNroadLabel == 'LEFT_TURN'):
        return 1
    elif (NNroadLabel == 'STRAIGHT'):
        return 2
    elif (NNroadLabel == 'RIGHT_TURN'):
        return 3
    else:
        print 'ERROR : road use case not define'
     

"""
@function      : num2RoadLabel(imgArray,NNroadLabel = None)
@description   : reshape the image from a 3 quarter to a 2 quarter 120x320 image
                 according to the use case 
@param fileName: array of image or image alone (first shape to discriminate)
               : NNroadLabel : use case road to be use for this image resizing
@rtype         : None
@return        : None
"""         
def num2RoadLabel(num):
    if (num == 0):
        return 'IDLE'
    elif (num == 1):
        return 'LEFT_TURN'
    elif (num == 2):
        return 'STRAIGHT'
    elif (num == 3):
        return 'RIGHT_TURN'
    else:
        print 'ERROR : num for road use case not define'       


'''
@function      : gps2MainRoadLabel(pos)
@description   : return the label according to gps pos
@param fileName: gps pos
@rtype         : None
@return        : road MainNN label (CROSS_ROAD,IDLE,LEFT_TURN,RIGHT_TURN
'''       
def gps2MainRoadLabel(pos):
    if gpsEnable == False:
        return 'IDLE'

    # If pos is in circle with center (mapCenter_x, mapCenter_Y) and with radius=straight_radius
    # return STRAIGHT otherwise return IDLE
    dx = abs(pos[0]-mapCenter_x)
    dy = abs(pos[1]-mapCenter_Y)
    
    if dx**2 + dy**2 <= straightRadius**2 : 
        return 'STRAIGHT'

    return 'IDLE'

       
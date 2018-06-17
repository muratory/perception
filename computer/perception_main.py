import cv2
import numpy as np
import threading
import Queue
import time
from ctypes import *
import math
import random


#libraries from deep drive
from commonDeepDriveDefine import *
from KeyboardThread import *
from VideoThread import *




####################################### Darknet part ##############################

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./cfgDarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict_p
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

make_image = lib.make_image
make_image.argtypes = [c_int,c_int,c_int]
make_image.restype = IMAGE


detect = lib.network_predict_p
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

load_image_cv = lib.load_image_cv
load_image_cv.argtypes = [c_char_p, c_int]
load_image_cv.restype = IMAGE

ipl_to_image = lib.ipl_to_image
ipl_to_image.argtypes = [c_void_p]
ipl_to_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res




####################################### Deep Drive Perception Thread ##############################


class PerceptionThread(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Video Stream Client Thread
        self.sctVideoCarStream = VideoThread()
        self.sctVideoCarStream.name = 'sctVideoCarStream'
        self.sctVideoCarStream.start()

        # create Keyboard Thread
        self.keyboardThread = keyboardThread(0,0)
        self.keyboardThread.name = 'Perception_Kb'
        self.keyboardThread.start()

        self.net = load_net("cfgDarknet/tiny-yolo-voc_demo.cfg", "cfgDarknet/tiny-yolo-voc_demo.weights", 0)
        self.meta = load_meta("cfgDarknet/dataFile_demo.data")

        #Perception server to provide detected object to client
        self.srvPerception = serverThread()
        self.srvPerception.name = 'srvPerception'
        self.srvPerception.start()
        
        
    def ConnectClient(self):
        # loop until all client connected
        videoCarClientConnected = False
        perceptionServerConnected = False
        

        # launch connection thread for all client
        if videoCarClientEnable == True:
            self.sctVideoCarStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_CAR_SERVER) + '/?action=stream'))


        if perceptionEnable == True:
            self.srvPerception.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_PERCEPTION_SERVER))

            
        while (videoCarClientConnected != videoCarClientEnable):
            # wait for .5 second before to check
            time.sleep(0.5)

            if (videoCarClientConnected != videoCarClientEnable):
                try:
                    reply = self.sctVideoCarStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoCarClientConnected = True
                        print 'Video stream server connected'
                except Queue.Empty:
                    print 'Video Client not connected'


            try:
                reply = self.keyboardThread.reply_q.get(False)
                if reply.type == ClientReply.SUCCESS:
                    if reply.data == 'exit':
                        return False
            except Queue.Empty:
                time.sleep(0.5)
                pass
            
        #otherwise it is ok and all is connected
        return True
    
    def run(self):

        lastKeyPressed = 0
        
        cv2.namedWindow('Perception')
        cv2.moveWindow('Perception', 250, 0)
        
        # initial steer command set to stop
        try:
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for perception'
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))

            lastFrameTime    = time.time()
            fpsMeasure =  np.zeros(30, dtype=np.float)
            fpsMeasureIdx=0
            
            while True:
                ############################# Manage IMAGE from car Camera ###############
                try:
                    # try to see if image ready for car vision
                    replyVideo = self.sctVideoCarStream.reply_q.get(False)

                    if replyVideo.type == ClientReply.SUCCESS:

                        # decode jpg into array
                        i = cv2.imdecode(np.fromstring(self.sctVideoCarStream.lastImage, dtype=np.uint8),-1)
                        
                        #write imaritege for other thread 
                        cv2.imwrite('frame.jpg',i)
                        
                        #measure fps
                        timeNow = time.time()
                        fpsMeasure[fpsMeasureIdx] = timeNow-lastFrameTime
                        fpsMeasureIdx+=1
                        if fpsMeasureIdx >=30 :
                            fpsMeasureIdx = 0
                        fps = 1/(np.sum(fpsMeasure)/30)
                        lastFrameTime=timeNow

                        
                        ############################# Perception ###############
                        r = detect(self.net, self.meta, 'frame.jpg')

                
                        if len(r)>0:
                            #print r
                            for element in r:
                                classObject,pb,bb = element
                
                                #print bb
                                x,y,w,h=bb
                                x1 = int(x-w/2)
                                y1 = int(y-h/2)
                                x2 = int(x+w/2)
                                y2 = int(y+h/2)
                                
                                kernel=(21,21)
                                
                                if classObject == 'yield':
                                    cv2.rectangle(i, (x1, y1), (x2,y2), (0,0,255), 2)
                                    coefFocal = 1                          
                                elif classObject == 'car':
                                    cv2.rectangle(i, (x1, y1), (x2,y2), (0,0,255), 2)
                                    coefFocal = 5
                                elif classObject == 'stop' :
                                    cv2.rectangle(i, (x1, y1), (x2,y2), (0,0,255), 2)
                                    coefFocal = 1
                                elif classObject == 'renault' :
                                    color = 255*np.random.rand(3)
                                    cv2.rectangle(i, (x1, y1), (x2,y2), color, 2)
                                    coefFocal=1
                                else:
                                    cv2.rectangle(i, (x1, y1), (x2,y2), (255,255,0), 2)   
                                    coefFocal = 3                             
             
                                #send object when detected 
                                distObj = coefFocal*3000/int(((x2-x1)+(y2-y1)))
                                
                                #write object class in black
                                cv2.putText(i,classObject+' '+str(distObj)+'cm',(x1+1,y1+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1)

                                '''
                                if distObj < 30.0:
                                    print 'send Object %s %s'%(classObject,distObj)
                                '''
                                
                                #filtering condition before to send the object to path control
                                #if classObject != 'renault' and distObj <40.0 :
                                if classObject != 'renault' :
                                    self.srvPerception.cmd_q.put(ClientCommand(ClientCommand.SEND, classObject+','+str(distObj)))
                            

                        
                        #after detection you can increase image size
                        i=cv2.pyrUp(i)
                        
                        #we have time to display some stuff
                        cv2.putText(i, 'Perception FPS  = ' + str(int(fps)), (0,20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)
                                                
                        #display the car vision and info
                        cv2.imshow('Perception', i)
                        
                        # check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print 'Error getting image :' + str(replyVideo.data)
                        break
                    
                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass

        

                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        # new keyboard input found
                        keyPressed = reply.data
                        strText = keyPressed

                        if keyPressed == 'exit':
                            sourceAngle = 0
                            turn_angle = 0
                            # get out of the loop
                            break
                        elif keyPressed == 'help':
                            strText = 'Key:Q'

                        else:
                            # key not known display error
                            self.keyboardThread.displayText()
                            strText=''

                        if strText != '':
                            self.keyboardThread.displayText(strText)
                            
                        # record lastkey that can be use for consecutive command action
                        lastKeyPressed = keyPressed

                    else:
                        print 'Error getting keyboard input :' + str(reply.data)
                        break
                except Queue.Empty:
                    # queue empty most of the time because keyboard not hit
                    pass

                
                        
        finally:
            print 'ending Perception'
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.srvPerception.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPerception.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            # and make sure all of them ended properly
            self.sctVideoCarStream.join()
            self.keyboardThread.join()
            self.srvPerception.join()
            print 'Perception Done'
            

if __name__ == '__main__':
    # create Deep drive thread and strt
    PerceptionThread = PerceptionThread()
    PerceptionThread.name = 'PerceptionThread'

    # start
    PerceptionThread.start()

    PerceptionThread.join()
    print 'end'

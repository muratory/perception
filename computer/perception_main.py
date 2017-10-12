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
from SensorThread import *
from gpsClientThread import *



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
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
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


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
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

        # create Sensor Client Thread
        self.sctCarSensor = SensorThread()
        self.sctCarSensor.name = 'SensorThread'
        self.sctCarSensor.start()

        # create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'car_main_kb'
        self.keyboardThread.start()


        #create Gps Thread
        self.sctGps = gpsClientThread()
        self.sctGps.name = 'GpsClientThread'
        self.sctGps.start()

        self.net = load_net("cfgDarknet/tiny-yolo-voc.cfg", "cfgDarknet/tiny-yolo-voc.weights", 0)
        self.meta = load_meta("cfgDarknet/voc.data")

        #test with camera
        self.cap = cv2.VideoCapture(0)

    def ConnectClient(self):
        # loop until all client connected
        videoCarClientConnected = False
        sensorClientConnected = False
        gpsClientConnected = False

        # launch connection thread for all client
        if videoCarClientEnable == True:
            self.sctVideoCarStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_CAR_SERVER) + '/?action=stream'))
            
            
        if sensorClientEnable == True:
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_SENSOR_SERVER))

        if gpsEnable == True:
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_GPS_FIX_SERVER))
            
        while ((videoCarClientConnected != videoCarClientEnable) or
                (sensorClientConnected != sensorClientEnable) or
                (gpsClientConnected != gpsEnable)):

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

            if (sensorClientConnected != sensorClientEnable):
                try:
                    reply = self.sctCarSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sensorClientConnected = True
                        print 'Sensor server connected'
                except Queue.Empty:
                    print 'Sensor Client not connected'

                    
            if (gpsClientConnected != gpsEnable):
                try:
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        gpsClientConnected=True
                        print 'Gps server connected'
                except Queue.Empty:
                    print 'Gps Client not connected' 
                    
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
        
        # initial steer command set to stop
        try:
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for perception'
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, VEHICULE_NAME))

            lastFrameTime    = 0
            fpsMeasure =  np.zeros(30, dtype=np.int)
            fpsMeasureIdx=0
            
            while True:
                '''
                ############################# Manage IMAGE from car Camera ###############
                try:
                    # try to see if image ready for car vision
                    replyVideo = self.sctVideoCarStream.reply_q.get(False)

                    if replyVideo.type == ClientReply.SUCCESS:

                        # decode jpg into array
                        i = cv2.imdecode(np.fromstring(self.sctVideoCarStream.lastImage, dtype=np.uint8),-1)
                        
                        #write imaritege for other thread 
                        cv2.imwrite('frame.png',i)
                        
                        #measure fps
                        timeNow = time.time()
                        fpsMeasure[fpsMeasureIdx] = timeNow-lastFrameTime
                        fpsMeasureIdx+=1
                        if fpsMeasureIdx >=30 :
                            fpsMeasureIdx = 0
                        fps = np.sum(fpsMeasure)/30
                        
                        #we have time to display some stuff
                        cv2.putText(i, 'Perception FPS  = ' + str(fps), (0,IMAGE_PIXELS_Y/2 + 0), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        
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
                '''
                
                ############################# test with camera ###############
                ret, frame = self.cap.read()
                
                cv2.imwrite('frame.png',frame)
                
                r = detect(self.net, self.meta, 'frame.png')
        
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
                        
                        if classObject == 'person':
                            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,0,255), 2)
                            #blur half upper part
                            sub_rec = frame[y1:y1+(y2-y1)/2,x1:x2]
                            if sub_rec.shape[0] > 21 and sub_rec.shape[1] > 21:
                                sub_rec = cv2.blur(sub_rec,kernel,35)
                                #print 'p =',sub_rec.shape
                                frame[y1:y1+(y2-y1)/2,x1:x2] = sub_rec
                        
                        elif classObject == 'car':
                            cv2.rectangle(frame, (x1, y1), (x2,y2), (255,0,0), 2)
                            #blur half upper part
                            sub_rec = frame[y1+(y2-y1)/2:y2,x1:x2]
                            if sub_rec.shape[0] > 21 and sub_rec.shape[1] > 21:
                                sub_rec = cv2.blur(sub_rec,kernel,35)
                                #print 'c =',sub_rec.shape
                                frame[y1+(y2-y1)/2:y2,x1:x2] = sub_rec
                        else :
                            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)
                            
                        #write object class in black
                        cv2.putText(frame,classObject+' prob='+str(int(pb*100)),(x1+1,y1+10),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
                                    
                            
                            
                # Display the resulting frame
                cv2.imshow('frame',frame)
                # check if we want to stop autonomous driving
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            



        
                ############################# Get Sensor value ###############
                try:
                    # try to see if image ready
                    reply = self.sctCarSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        if (reply.data < STOP_DISTANCE):
                            print 'sensor value = ' + str(reply.data)
                            if (inputAngle == 0) :
                                self.sctCarSteering.cmd_q.put(ClientCommand(
                                    ClientCommand.SEND, ('STEER_COMMAND', 'stop')))
                        elif startCar == 1 :
                            #move forward the car only if we ordered to start it previously 
                            self.sctCarSteering.cmd_q.put(ClientCommand(
                                ClientCommand.SEND, ('STEER_COMMAND', 'forward')))

                    else:
                        print 'Error getting Sensor :' + str(reply.data)
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
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctCarSensor.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            # and make sure all of them ended properly
            self.sctVideoCarStream.join()
            self.sctCarSensor.join()
            self.keyboardThread.join()
            self.sctGps.join()
            print 'Deep Driver Done'
            
                
            # When everything done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # create Deep drive thread and strt
    PerceptionThread = PerceptionThread()
    PerceptionThread.name = 'PerceptionThread'

    # start
    PerceptionThread.start()

    PerceptionThread.join()
    print 'end'
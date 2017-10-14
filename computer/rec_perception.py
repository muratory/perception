import cv2
import numpy as np
import math
import threading
import Queue
import time
from commonDeepDriveDefine import *
from KeyboardThread import *
from SteerThread import *
from VideoThread import *
from SensorThread import *

sensorClientEnable = False
class CollectTrainingData(threading.Thread):
    
    def __init__(self):

        #call init
        threading.Thread.__init__(self)
   
        #create Video Stream Client Thread
        self.sctVideoStream = VideoThread()
        self.sctVideoStream.name = 'VideoSocketThread'
        self.sctVideoStream.start()
        
        #create Steer Client Thread
        self.sctSteer = SteerThread()
        self.sctSteer.name = 'SteerSocketThread'
        self.sctSteer.start()

        #create Sensor Client Thread
        self.sctSensor = SensorThread()
        self.sctSensor.name = 'SensorSocketThread'
        self.sctSensor.start()

        #create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'car_calib_Kb'
        self.keyboardThread.start()
        


    def ConnectClient(self):
        # loop until all client connected
        videoClientConnected = False
        steerClientConnected = False
        sensorClientConnected = False
        
        #launch connection thread for all client
        self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CONNECT, 'http://' + CAR_IP + ':' + str(PORT_VIDEO_CAR_SERVER) + '/?action=stream'))
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_STEER_SERVER))
        self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_SENSOR_SERVER))

        while (videoClientConnected != videoClientEnable) :

            #wait for .5 second before to check 
            time.sleep(0.5)
            
            if (videoClientConnected != videoClientEnable):
                try:
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoClientConnected=True
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

    def run(self):
        
        saved_frame = 0
        total_frame = 0
        lastTotalAngle = 0
        totalRecordTime=0
        frame = 1
        record = 0
        bytes=''
        turn_angle = 0
        totalAngle = 0
        lastkeypressed = 0
        recordTime = 0
        totalRecordTime = 0
                     
            
        #Send speed to car 
        
        # stream video frames one by one
        try:         
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            print 'Start Main Thread to calibrate car'
            
            #start receiver thread client to receive continuously data
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start keyboard thread to get keyboard inut
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #init time
            lastFrameTime = time.time()
            lastSteerTime = lastFrameTime


            idx = 0;
            while True:
                idx+=1
                try:
                    # check queue success for image ready
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        i = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),-1)
                        #img_out = cv2.cvtColor(i, cv2.COLOR_BGR2YCrCb)
                        #print 'value = ',img_out[240/2][320/2]
                        
                        cv2.imshow('image', i)
                        #cv2.imshow('roi_image', flipped_roi)
                        #cv2.imshow('image', i)

                        #check if we want to stop autonomous driving
                        if cv2.waitKey(500) & 0xFF == ord('q'):
                            break
                        
                        cv2.imwrite('./images/image_'+str(idx)+'.JPEG',i)
                        
                        
                    else:
                        print 'Error getting image :' + str(reply.data)
                        break
                    
                except Queue.Empty:
                    #queue empty most of the time because image not ready
                    pass

                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #new keyboard input found
                        keyPressed = reply.data
                        strText = keyPressed
                        
                        if keyPressed == 'exit':
                            record = 0
                            turn_angle = 0
                            if recordTime != 0:
                                totalRecordTime += (time.time() - recordTime)
                            #get out of the loop
                            break
                        elif keyPressed == 'help':
                            strText = 'Key:Q,Arrow,+,-,space'
                        
                        elif keyPressed == 'right':
                            turn_angle = STEP_CAPTURE
                            
                        elif keyPressed == 'left':
                            turn_angle = -STEP_CAPTURE

                        elif keyPressed == 'up':
                            if (self.sctSteer.steerCommand == 'backward') :
                                # stop if we were in back before 
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                                #but no record since we are stop
                            else:
                                #in all other case, go forward and record
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))

                        elif keyPressed == 'down':
                            if (self.sctSteer.steerCommand == 'stop') :
                                # go back if we were stop before 
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','backward')))
                            else:
                                #in all other case, just stop
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))   
                            
                        elif keyPressed == 'minus':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',self.sctSteer.speed - 5)))
                            
                        elif keyPressed == 'plus':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',self.sctSteer.speed + 5)))
                            
                        elif keyPressed == 'space':
                            record = 1
                            
                        elif keyPressed == 'none':
                            turn_angle = 0
                            if lastkeypressed == 'space':
                                #this was a free space record then we canstop record
                                record = 0         
                            strText=''                                                
                        else :
                            # key not known display error
                            self.keyboardThread.displayText()
                            strText=''

                        if strText != '':
                            self.keyboardThread.displayText(strText)
                            
                        # record lastkey that can be use for consecutive command action
                        lastkeypressed = keyPressed
                            
                    else:
                        print 'Error getting keyboard input :' + str(reply.data)
                        break             
                except Queue.Empty:
                    #queue empty most of the time because keyboard not hit
                    pass
                

                #get time for steer command and apply it if done
                timeNow = time.time()
                if timeNow > (lastSteerTime + STEERING_KEYBOARD_SAMPLING_TIME):
                    #it s time to update steer command
                    totalAngle += turn_angle
                    if totalAngle >= MAX_ANGLE:
                        totalAngle = MAX_ANGLE
                    elif totalAngle <= MIN_ANGLE:
                        totalAngle = MIN_ANGLE                        
                    self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',totalAngle)))
                    lastSteerTime = timeNow
                    if lastTotalAngle != totalAngle:
                        print 'turn_angle = ',totalAngle
                        lastTotalAngle = totalAngle
                
        
        finally:
                #stop and close all client and close them
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
                
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                #let 1 second for process to close
                time.sleep(2)
                self.sctVideoStream.join()
                self.sctSteer.join()
                self.sctSensor.join()
                self.keyboardThread.join()

if __name__ == '__main__':

    #create Deep drive thread and strt
    DDCollectData = CollectTrainingData()
    DDCollectData.name = 'DDriveThread'
    
    #start
    DDCollectData.start()

    DDCollectData.join()

    print 'end'

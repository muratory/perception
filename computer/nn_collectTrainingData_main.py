import cv2
import numpy as np
import math
import threading
import Queue
import time
from commonDeepDriveDefine import *
from nnCommonDefine import *
from KeyboardThread import *
from SteerThread import *
from VideoThread import *
from SensorThread import *
from gpsClientThread import *


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
        self.keyboardThread.name = 'nn_collectTraining_Kb'
        self.keyboardThread.start()
        
        #create Gps Thread
        self.sctGps = gpsClientThread()
        self.sctGps.name = 'GpsClientThread'
        self.sctGps.start()
        
        self.portVideo  = PORT_VIDEO_CAR_SERVER
        self.dataFolder = "training_data"

        

    def ConnectClient(self):
        # loop until all client connected
        videoClientConnected = False
        steerClientConnected = False
        sensorClientConnected = False
        gpsClientConnected = False
        
        #launch connection thread for all client
        self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CONNECT, 'http://' + CAR_IP + ':' + str(self.portVideo) + '/?action=stream'))
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_STEER_SERVER))
        self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_SENSOR_SERVER))
        self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_GPS_FIX_SERVER))

        while ( (videoClientConnected != videoClientEnable) or
                (steerClientConnected != steerClientEnable) or
                (sensorClientConnected != sensorClientEnable) or
                (gpsClientConnected != gpsEnable)):

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

            if (steerClientConnected != steerClientEnable):
                try:
                    reply = self.sctSteer.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        steerClientConnected=True
                        print 'Steer server connected'
                except Queue.Empty:
                    print 'Steer Client not connected'

            if (sensorClientConnected != sensorClientEnable):
                try:
                    reply = self.sctSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sensorClientConnected=True
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

    def run(self):
        
        saved_frame = 0
        total_frame = 0
        lastTotalAngle = 0
        frame = 1
        record = 0
        turn_angle = 0
        totalAngle = 0
        lastkeypressed = 0
        recordTime = 0
        totalRecordTime = 0
        gpsPosition = (0,0)
        gpsAngle = 0
        itineraryStarted = False
        NNroadUseCase = 'NONE'
        
        forceManualLabel = not(gpsEnable)

            
        #Send speed to car 
        print 'Enter main thread to collect data'
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',45)))
        #initial steer command set to stop and no angle
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
        
        #array to be use for labelling 
        image_array = np.zeros((1,IMAGE_PIXELS_Y, IMAGE_PIXELS_X), dtype=np.uint8)            
        steerAngle_label_array = np.zeros(1, dtype=np.uint8)
        NNroadUseCase_label_array = np.zeros(1, dtype=np.uint8)
        gpsPosition_label_array = np.zeros((1,2), dtype=np.uint16)
        
        
        # stream video frames one by one
        try:         
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return

            print 'Start Main Thread to collect image'
            #start receiver thread client to receive continuously data
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
            #start Sensor 
            self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start gps to receive fix
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #init time
            lastFrameTime = time.time()
            lastSteerTime = lastFrameTime

            while True:
                try:
                    # check queue success for image ready
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        i = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),-1)
                        
                        #keep only gray image for NN
                        image2nn = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                        
                        #write speed
                        cv2.putText(i, 'speed = ' + str(self.sctSteer.speed), (0,IMAGE_PIXELS_Y/2), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        
                        #Select if you want to override GPS labelling by manual labelling
                        if forceManualLabel == False:
                            #gps2MainRoadLabel only return straight or idle .. other nn has to be train manually
                            NNroadUseCase = gps2MainRoadLabel(gpsPosition)
                            cv2.putText(i, 'GPS control', (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        else:
                            cv2.putText(i, 'MANUAL control', (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                            if NNroadUseCase == 'NONE':
                                cv2.putText(i, 'select NN to train (l)eft, (r)ight, ,(s)straight, (i)dle', (0,IMAGE_PIXELS_Y/2 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                                cv2.putText(i, '(l)eft, (r)ight, ,(s)straight, (i)dle', (0,IMAGE_PIXELS_Y/2 + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)                            

                        
                        cv2.putText(i, 'NNRoadUseCase = ' + NNroadUseCase, (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'GPSRoadUseCas = ' + gps2MainRoadLabel(gpsPosition), (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        
                        '''
                        #draw ROAD selection 
                        #for NN roadNN, the image recorded depend on the label road label used
                        if self.NNroadUseCase == 'RIGHT_TURN':
                            cv2.rectangle(i,(IMAGE_PIXELS_X/2,0),(IMAGE_PIXELS_X,IMAGE_PIXELS_Y),(0,0,255),1)
                        elif self.NNroadUseCase == 'LEFT_TURN':
                            cv2.rectangle(i,(IMAGE_PIXELS_X/4,0),(IMAGE_PIXELS_X*3/4,IMAGE_PIXELS_Y/2),(0,0,255),1)
                            cv2.rectangle(i,(IMAGE_PIXELS_X/2,IMAGE_PIXELS_Y/2),(IMAGE_PIXELS_X,IMAGE_PIXELS_Y),(0,0,255),1)
                        elif self.NNroadUseCase == 'IDLE' or self.NNroadUseCase == 'CROSS_ROAD' :
                            cv2.rectangle(i,(0,0),(IMAGE_PIXELS_X,IMAGE_PIXELS_Y/2),(0,0,255),1)
                        '''
                        
                        #draw red Circle if record
                        if (record == 1) :
                            cv2.circle(i,(IMAGE_PIXELS_X-20,20),10,(0,0,255),-1)
                        
                        #add Gps Position 
                        cv2.putText(i, 'GPS position =' + str(gpsPosition), (0,IMAGE_PIXELS_Y/2 + 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        
                        cv2.imshow('Image', i)
                        
                        #check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        #expand dimension of image to allow the stack later
                        temp_array = np.expand_dims(image2nn, axis=0)
                        
                        frame += 1
                        total_frame += 1                    
                        event_handled = 0
                        
                    else:
                        print 'Error getting image :' + str(reply.data)
                        break
                    
                except Queue.Empty:
                    #queue empty most of the time because image not ready
                    pass
                    
                ############################# Get Gps angle value ###############
                try:

                    # try to see if image ready
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                            #fix has been receive . process it
                            vPosX, vPosY, vSpeed, vOrient, vName = reply.data.split(',')
                            #print 'Received FIX for vehicle ',vName, ' x=', vPosX, ' y=', vPosY
                            if (vName == VEHICULE_NAME):
                                ############### SEND value to Graph slam ############
                                gpsPosition = (int(vPosX),int(vPosY))                            
                    else:
                        print 'Error getting Gps value :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass
                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #new keyboard input found
                        keyPressed = reply.data
                        print 'key Pressed = ' , keyPressed
                        
                        if keyPressed == 'exit':
                            record = 0
                            turn_angle = 0
                            if recordTime != 0:
                                totalRecordTime += (time.time() - recordTime)
                            #get out of the loop
                            break
                        
                        elif keyPressed == 'right':
                            turn_angle = STEP_CAPTURE
                            
                        elif keyPressed == 'left':
                            turn_angle = -STEP_CAPTURE

                        elif keyPressed == 'up':
                            if (self.sctSteer.steerCommand == 'backward') :
                                # stop if we were in back before 
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                                #but no record since we are stop
                                record = 0 
                            else:
                                #in all other case, go forward and record
                                record = 1
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))

                        elif keyPressed == 'down':
                            record = 0
                            if (self.sctSteer.steerCommand == 'stop') :
                                # go back if we were stop before 
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','backward')))
                            else:
                                #in all other case, just stop
                                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                            
                        elif keyPressed == 'space':
                            record = 1

                        elif (keyPressed == 'RIGHT_TURN') or (keyPressed == 'STRAIGHT') or (keyPressed == 'LEFT_TURN') or (keyPressed == 'IDLE'):
                            NNroadUseCase = keyPressed
                            forceManualLabel = True
                            if lastkeypressed == keyPressed:
                                forceManualLabel = False
                            
                        elif keyPressed == 'minus':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',self.sctSteer.speed - 5)))
                            
                        elif keyPressed == 'plus':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',self.sctSteer.speed + 5)))

                                
                        elif keyPressed == 'none':
                            turn_angle = 0
                            if lastkeypressed == 'space':
                                #this was a free space record then we canstop record
                                record = 0                                                         
                        else :
                            #none expeted key is pressed
                            print 'Error , this key does not exist !!!'
                            
                        # record lastkey that can be use for consecutive command action
                        if keyPressed != 'none':
                            lastkeypressed = keyPressed
                            
                    else:
                        print 'Error getting keyboard input :' + str(reply.data)
                        break             
                except Queue.Empty:
                    #queue empty most of the time because keyboard not hit
                    pass
                
                #See now if we have to record or not the frame into vstack memory
                timeNow = time.time()
                if (record == 1):
                    #start recording time if all condition are met
                    if recordTime == 0:
                        if ((NNroadUseCase != 'NONE' and forceManualLabel == True) or ((gpsEnable == True) and (forceManualLabel == False))):
                            recordTime = time.time()
                        else:
                            print 'WARNING : to record please select a road use case with l,r,i,c...'
                            record = 0
                            continue

                    
                    #check if this is time to record a frame
                    if  timeNow > (lastFrameTime + FPS_RECORD_TIME):
                        #record image and labels
                        saved_frame += 1                   
                        angle = totalAngle

                        image_array = np.vstack((image_array, temp_array))
                        NNroadUseCase_label_array = np.vstack((NNroadUseCase_label_array, np.array([NNroadLabel2Num(NNroadUseCase)])))
                        steerAngle_label_array = np.vstack((steerAngle_label_array, np.array([angle])))
                        if gpsEnable:
                            gpsPosition_label_array = np.vstack((gpsPosition_label_array, np.array(gpsPosition)))
                         
                        lastFrameTime = timeNow
                else:
                    #record the time if recorTime exist
                    if recordTime != 0:
                        totalRecordTime += (timeNow - recordTime)
                        recordTime = 0

                #get time for steer command and apply it if done
                timeNow = time.time()
                if timeNow > (lastSteerTime + STEERING_KEYBOARD_SAMPLING_TIME):
                    #it s time to update steer command
                    totalAngle += turn_angle
                        
                    totalAngle += turn_angle
                    if totalAngle >= MAX_ANGLE:
                        totalAngle = MAX_ANGLE
                    elif totalAngle <= MIN_ANGLE:
                        totalAngle = MIN_ANGLE              
                    #print 'send to steer ',totalAngle     
                    self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',totalAngle)))
                    lastSteerTime = timeNow
                    if lastTotalAngle != totalAngle:
                        print 'turn_angle = ',totalAngle
                        lastTotalAngle = totalAngle
                
            if totalRecordTime !=0:            

                # Build file name based on date/time
                timestr  = time.strftime("%Y%m%d-%H%M%S")
                fileName = self.dataFolder + '/trainingSet_' + timestr
                    
                if len(image_array) > 1:
                    # save training images for NN one (main NN to select road case)
                    image_array = image_array[1:, :]
                    steerAngle_label_array = steerAngle_label_array[1:, :]
                    NNroadUseCase_label_array = NNroadUseCase_label_array[1:, :]
                    gpsPosition_label_array = gpsPosition_label_array[1:, :]
                    
                    print 'road Label distribution ONE'
                    print np.histogram(NNroadUseCase_label_array,bins=range(0,5))
                    print 'image shape       =', image_array.shape
                    print 'steer angle shape =', steerAngle_label_array.shape
                    print 'road label shape  =', NNroadUseCase_label_array.shape
                    print 'gps label shape   =', gpsPosition_label_array.shape
                    
                    # save training data as a numpy file
                    np.savez(fileName, train=image_array, steerAngle_label_array=steerAngle_label_array, NNroadUseCase_label_array=NNroadUseCase_label_array, gpsPosition_label_array=gpsPosition_label_array)

                print 'Total frame:', total_frame
                print 'Saved frame:', saved_frame , ' in ', totalRecordTime, ' seconds'
                print 'Dropped frame', total_frame - saved_frame

        
        finally:
                #stop and close all client and close them
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.sctGps.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                #let 1 second for process to close
                time.sleep(2)
                self.sctVideoStream.join()
                self.sctSteer.join()
                self.sctSensor.join()
                self.sctGps.join()
                self.keyboardThread.join()

if __name__ == '__main__':

    #create Deep drive thread and strt

    DDCollectData = CollectTrainingData()
    
    #name
    DDCollectData.name = 'DDCollectData'
    
    #start
    DDCollectData.start()
    
    DDCollectData.join()
    
    print 'end'

import cv2
import numpy as np
import threading
import Queue
import time

#libraries from deep drive
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from KeyboardThread import *
from VideoThread import *
from gpsClientThread import *
from graphSlamThread import *








####################################### Deep Drive Thread ##############################





class pathControl(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Video Stream Client Thread used mainly for Land Mark and obj detec. 
        self.sctVideoCarStream = VideoThread()
        self.sctVideoCarStream.name = 'sctVideoCarStream'
        self.sctVideoCarStream.start()


        # create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'pathControl_Kb'
        self.keyboardThread.start()
        
        #create GPS path control server to provide steer Angle based Graphslam and GPS pos
        self.srvPathControlSteering = serverThread()
        self.srvPathControlSteering.name = 'srvPathControlSteering'
        self.srvPathControlSteering.start()
        
        #server which provide moveOrder to the car such as :
        #FORWARD, STOP, SPEED=XXX, IDLE, STRAIGHT, LEFT, RIGHT, CHANGE_RIGHT, CHANGE_LEFT
        self.srvPathControlCommand = serverThread()
        self.srvPathControlCommand.name = 'srvPathControlCommand'
        self.srvPathControlCommand.start()

        
        #create Gps Thread that receive image from camera and detect cars
        self.sctGps = gpsClientThread()
        self.sctGps.name = 'GpsClientPathControl'
        self.sctGps.start()

        #create GraphSlam Thread to compute slam
        self.graphSlamThread = graphSlamThread()
        self.graphSlamThread.name = 'GraphSlamThread'
        self.graphSlamThread.start()
        


    def ConnectClient(self):
        # loop until all client connected
        videoCarClientConnected = False
        sctGpsConnected = False
        gpsServerFixConnected = False
        PathControlCommandServerConnected = False
        pathControlSteeringServerConnected = False
               

        # launch connection thread for all client
        if videoCarClientEnable == True:
            self.sctVideoCarStream.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + CAR_IP + ':' +
                str(PORT_VIDEO_CAR_SERVER) + '/?action=stream'))
                    
        if gpsEnable == True:
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_GPS_FIX_SERVER))
            
                
        if pathControlSteeringEnable == True:
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_PATH_CONTROL_STEERING_SERVER))
            
        if pathControlCommandEnable == True:
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_PATH_CONTROL_COMMAND_SERVER))


        while ((videoCarClientConnected != videoCarClientEnable) or
                (pathControlSteeringServerConnected != pathControlSteeringEnable) or
                (PathControlCommandServerConnected != pathControlCommandEnable) or
                (sctGpsConnected != gpsEnable)):

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
                    
            if (pathControlSteeringServerConnected != pathControlSteeringEnable):
                try:
                    reply = self.srvPathControlSteering.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        pathControlSteeringServerConnected=True
                        print 'steering pathControl server connected'
                except Queue.Empty:
                    print 'steering pathControl server not connected' 
                    
            if (PathControlCommandServerConnected != pathControlCommandEnable):
                try:
                    reply = self.srvPathControlCommand.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        PathControlCommandServerConnected=True
                        print 'pathControl command server connected'
                except Queue.Empty:
                    print 'pathControl command not connected' 
            
            if (sctGpsConnected != gpsEnable):
                try:
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sctGpsConnected=True
                        print 'Gps fix client connected'
                except Queue.Empty:
                    print 'Gps fix Client not connected' 
                    
            try:
                reply = self.keyboardThread.reply_q.get(False)
                if reply.type == ClientReply.SUCCESS:
                    if reply.data == 'exit':
                        return False
            except Queue.Empty:
                time.sleep(0.5)
                pass
                    

    def run(self):
        lastKeyPressed = 0
        gpsPosition = (0,0)
        gpsSpeed=0
        #path control command
        pathControlCommand = 'IDLE'
        lastPastControlCommand = 'NONE'
        
        #path control speed
        pathControlSpeed = INITIAL_CAR_SPEED
        lastPastControlSpeed = 0
        lastCommandTime = time.time()

        if graphSlamEnable:
            cv2.namedWindow('GraphSlam')
 
        steerPathAngle = 0 #angle coming from Position module and graph slam ...
        

        # initial steer command set to stop
        try:
            
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for path Control '
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))

            while True:
                ################# Manage IMAGE from car Camera that should be use for LM and Obj detec ###############
                try:
                    # try to see if image ready for car vision
                    replyVideo = self.sctVideoCarStream.reply_q.get(False)
                    if replyVideo.type == ClientReply.SUCCESS:

                        # decode jpg into array
                        i = cv2.imdecode(np.fromstring(self.sctVideoCarStream.lastImage, dtype=np.uint8),-1)

                        cv2.putText(i, 'Path SteerAngle  = ' + str(steerPathAngle), (0,IMAGE_PIXELS_Y/2), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'GPS position =' + str(gpsPosition), (0,IMAGE_PIXELS_Y/2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'pathControlCommand =' + lastPastControlCommand, (0,IMAGE_PIXELS_Y/2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(i, 'pathControlSpeed =' + str(lastPastControlSpeed), (0,IMAGE_PIXELS_Y/2 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) 
                        cv2.putText(i, 'GPS speed mm/s  =' + str(gpsSpeed), (0,IMAGE_PIXELS_Y/2 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        
                        #show green line steering angle
                        showLabel(steerPathAngle,'PathControlVision', i)
                        
                        #display the car vision and info
                        cv2.imshow('PathControlVision', i)
                        
                        # check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print 'Error getting image :' + str(replyVideo.data)
                        break
                    
                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass
                

                ################# Handle fix from GPS ###############
                try:
                    #check if car gps fix has been detected
                    reply = self.sctGps.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #fix has been receive . process it
                        vPosX, vPosY, vGpsSpeed, vOrient, vName = reply.data.split(',')
                        #print 'Received FIX for vehicle ',vName, ' x=', vPosX, ' y=', vPosY
                        if (vName == VEHICULE_NAME):
                            ############### SEND value to Graph slam ############
                            gpsPosition = (int(vPosX),int(vPosY),int(vGpsSpeed),float(vOrient))
                            gpsSpeed = int(vGpsSpeed)
                            
                            #send GPS info to slam if feqture enable
                            if int(vGpsSpeed) > 50:
                                if graphSlamEnable == True:
                                    self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, gpsPosition))

                except Queue.Empty:
                    # queue empty most of the time because image not ready
                    pass
                
                
                
                      
                #############################Handle gpraphSlam ###############
                #check now if the slamGraph Thread posted a new steeringAngle
                try:
                    # try to see if data ready
                    reply = self.graphSlamThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        steerPathAngle = int(reply.data)
                        print 'receive Angle from graphSlamThread = ' + str(steerPathAngle)
                         
                        # Send Angle to server
                        self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, steerPathAngle))
                                                            
                        #show graphslam result in main thread
                        if self.graphSlamThread.myUserInterface.GraphSlamImg != None:
                            cv2.imshow('GraphSlam',self.graphSlamThread.myUserInterface.GraphSlamImg)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    else:
                        print 'Error getting GraphSlam data :' + str(reply.data)
                        break

                except Queue.Empty:
                    # queue empty most of the time because data not ready
                    pass


                ############################# Handle pathControl Command 5speed + control ####################################     
                    
                ############################# Handle pathControl Speed ####################################
                #no tuning of speed for the moment, just return INITIAL one
                
                if pathControlSpeed < 0:
                    pathControlSpeed = 0
                elif pathControlSpeed > MAX_CAR_SPEED:
                    pathControlSpeed = MAX_CAR_SPEED

                if pathControlSpeed != lastPastControlSpeed:
                    lastPastControlSpeed = pathControlSpeed
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, 'SPEED='+str(pathControlSpeed)))
 
                                    
                ################## first send the control command if necessary #####################
                # for the moment only set pathControlCommand to say IDLE or STRAIGHT depending
                pathControlCommand = gps2MainRoadLabel(gpsPosition)
                             
                if pathControlCommand != lastPastControlCommand:
                    lastPastControlCommand = pathControlCommand
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, pathControlCommand))

            
            
             

                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        # new keyboard input found
                        keyPressed = reply.data
                        strText = keyPressed

                        if keyPressed == 'exit':
                            # get out of the loop
                            break
                        
                        elif keyPressed == 'help':
                            strText = 'Key:Q,P(lot),+,-'
                        
                        elif keyPressed == 'minus':
                            pathControlSpeed -= 50
                            
                        elif keyPressed == 'plus':
                            pathControlSpeed += 50
                            
                        elif keyPressed == 'PATH_CONTROL':
                            strText='Plot Slam'
                            #request with send to plot the graph
                            self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.SEND, ''))
                            
                        elif keyPressed == 'none':
                            strText=''

                            
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



                ######################## make sure to send command on all client because client could be new 
                # make sure to send the command at least every second
                timeNow = time.time()
                if timeNow > (lastCommandTime + 1.0):

                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, pathControlCommand))
                    self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.SEND, 'SPEED='+str(pathControlSpeed)))
                    self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.SEND, steerPathAngle))
                    
                    lastCommandTime = timeNow


                       
        finally:
            print 'ending path control Thread'

            # stop and close all client and close them
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoCarStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlSteering.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvPathControlCommand.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctGps.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.graphSlamThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
                                                       
            # and make sure all of them ended properly
            self.sctVideoCarStream.join()
            self.keyboardThread.join()
            self.srvPathControlSteering.join()
            self.srvPathControlCommand.join()
            self.graphSlamThread.join()
            print 'Path control Done'

if __name__ == '__main__':

    pathControl = pathControl()
    pathControl.name = 'pathControl'

    # start
    pathControl.start()

    pathControl.join()
    print 'end'

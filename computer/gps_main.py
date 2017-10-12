import cv2
import numpy as np
import threading
import Queue
import time
import pickle
import Gnuplot

#libraries from deep drive
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from KeyboardThread import *
from gpsFixThread import *

#step in mm expected before to record a new poitn for the Map
STEP_MAP_DISTANCE = 400


####################################### Deep Drive Thread ##############################


class gpsThread(threading.Thread):

    def __init__(self):
        # call init
        threading.Thread.__init__(self)

        # create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'gps_main_Kb'
        self.keyboardThread.start()
        
        
        #create Gps Thread that receive image from camera and detect cars
        self.gpsFixThread = gpsFixThread()
        self.gpsFixThread.name = 'GpsFixThread'
        self.gpsFixThread.start()

        #create Gps server that provide GPS fix for any client
        self.srvGpsFix = serverThread()
        self.srvGpsFix.name = 'GpsServerFix'
        self.srvGpsFix.start()


    def ConnectClient(self):
        # loop until all client connected
        gpsFixThreadConnected = False
        gpsServerFixConnected = False
               

        # launch connection thread for all client  
        if gpsEnable == True:
            self.gpsFixThread.cmd_q.put(ClientCommand(
                ClientCommand.CONNECT, 'http://' + GPS_VIDEO_IP + ':' +
                str(PORT_VIDEO_GPS_SERVER) + '/?action=stream'))
            
            self.srvGpsFix.cmd_q.put(ClientCommand(ClientCommand.CONNECT, PORT_GPS_FIX_SERVER))


        while ((gpsServerFixConnected != gpsEnable) or
                (gpsFixThreadConnected != gpsEnable)):

            # wait for .5 second before to check
            time.sleep(0.5)
            
            if (gpsFixThreadConnected != gpsEnable):
                try:
                    reply = self.gpsFixThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        gpsFixThreadConnected=True
                        print 'Gps Video client connected'
                except Queue.Empty:
                    print 'Gps Video Client not connected' 
                    

            if (gpsServerFixConnected != gpsEnable):
                try:
                    reply = self.srvGpsFix.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        gpsServerFixConnected=True
                        print 'Gps server for Fix connected'
                except Queue.Empty:
                    print 'Gps server for Fix not connected'
                     
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
        # initial steer command set to stop
        fixdata = {}
        index = 0  

        vName =''
        
        createMap = False
        
        firstGpsPositionReceived = False
        

        try:
            # start keyboard thread to get keyboard input
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))        
    
            # connect All client
            if self.ConnectClient() == False :
                return
            
            print 'Start Main Thread and sub Thread for path Control '
            self.gpsFixThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE, ''))


            while True:
                ################# Handle Image from GPS and get detection for all Cars   ###############
                try:
                    #check if car gps fix has been detected
                    reply = self.gpsFixThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #fix has been receive . process it
                        vPosX, vPosY, vSpeed,vOrient, vName = reply.data.split(',')
                        print 'Received FIX for vehicle ',vName, ' x=', vPosX, ' y=', vPosY, ' v=', vSpeed, 'Orient=', vOrient, ' t=', str(round(time.time() % 100,2))
                        
                        ############### SEND fix value to server ############
                        #vehicule gps fix detected . send to server for client waiting on it
                        self.srvGpsFix.cmd_q.put(ClientCommand(ClientCommand.SEND, reply.data))


                    if createMap == True:
                        if firstGpsPositionReceived == False:
                            #record a point
                            print 'record first point'
                            if vName != '':
                                print "INDEX ",index
                                print "NAME", vName
                                print "XPOS ",vPosX
                                print "YPOS ",vPosY
                                fixdata[index] = [index, int(vPosX), int(vPosY)]
                                index += 1
                                lastGpsPosition = (int(vPosX),int(vPosY))
                                firstGpsPositionReceived = True
                        else:
                            if distance(lastGpsPosition,(int(vPosX),int(vPosY))) > STEP_MAP_DISTANCE:
                                lastGpsPosition = (int(vPosX),int(vPosY))
                                print "INDEX ",index
                                print "NAME", vName
                                print "XPOS ",vPosX
                                print "YPOS ",vPosY
                                fixdata[index] = [index, int(vPosX), int(vPosY)]
                                index += 1
                             

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
                            # get out of the loop                                 
                            break
                        
                        elif keyPressed == 'help':
                            strText = 'Key:Q,Space(point),M(ap)'

                        elif keyPressed == 'none':
                            strText=''
                        
                        elif keyPressed == 'space':
                            strText='Record Point :' + vPosX + ' '+ vPosY
                            #record a point
                            if vName != '':
                                print "INDEX ",index
                                print "NAME", vName
                                print "XPOS ",vPosX
                                print "YPOS ",vPosY
                                fixdata[index] = [index, int(vPosX), int(vPosY)]
                                index += 1
                        
                        elif keyPressed == 'MAP':
                            if createMap == False:
                                createMap = True
                            else: 
                                createMap = False
                                                            
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
            
            if index > 0:
                print "record has been done, so save file and plot it :"
                data_pos = open("gps_plot.txt", "wb")
                pickle.dump(fixdata, data_pos)
                data_pos.close()
                
                #record in file the points for gnuplot use
                pngInput = open("gps_map.txt", "wb")
                
                for idx in range(0,index):
                    strPoint = str(fixdata[idx][0]) + ' ' + str(fixdata[idx][1]) + ' ' + str(fixdata[idx][2])
                    print strPoint
                    print >>pngInput,strPoint
                #close file
                pngInput.close()
                
                #create png 
		g = Gnuplot.Gnuplot()
		g.title('GPS plot')
		g.xlabel('X pos')
		g.ylabel('Y pos')
		g('set auto x')
		g('set xtics format ""')
		g('set x2tics')
		g('set yrange [2700:0]')
		g('set term png')
		g('set out "gps_map.png"')


		databuff = Gnuplot.File("./gps_map.txt", using='2:3:1 with labels offset 0.5,0.5 notitle axes x2y1')
		g.plot(databuff)              
            
            print 'ending GPS Thread'

            # stop and close all client and close them
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.gpsFixThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.gpsFixThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.srvGpsFix.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.srvGpsFix.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                                                       
            # and make sure all of them ended properly
            self.keyboardThread.join()
            self.gpsFixThread.join()
            self.srvGpsFix.join()
            print 'GPS Done'

if __name__ == '__main__':

    gps = gpsThread()
    gps.name = 'gpsThread'

    # start
    gps.start()

    gps.join()
    print 'end'

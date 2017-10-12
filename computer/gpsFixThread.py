import threading
import urllib
import Queue
import struct
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
from VideoThread import *
import argparse
import time
import imutils
import numpy as np
import sys
import pickle
from math import atan2

cam_gps_calib_file = "gps_camera_calib_folder/calibrate_gps_camera.p"
circuit_zone_of_interest_file = "gps_camera_calib_folder/undistort_gps_calibration.tpz"

debug_camera_trace = ''

list_of_v = list()

#Resize the camera input 
RESIZE = 0

mouseXY = (0,0)


def mousePosition(event,x,y,flags,param):

    if event == cv2.EVENT_MOUSEMOVE:
        global mouseXY
        mouseXY=(x,y)

    

def hls_thresh(img, thresh=(100, 255)):
        """
        Convert BGR to HLS and threshold to binary image using S channel
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output


def object_tracker(pts,frame,color,cam):
        debug_camera_trace=configuration.DEBUG_CAMERA_LEVEL

        # define the lower and upper boundaries of the "green" "blue"
        # object in the HSV color space, then initialize the
        # list of tracked points
        if color == 'BLUE':
            # http://colorizer.org/ for color mask computation
            hsv_colorlower=(110, 50, 50)
            hsv_colorupper=(130, 255, 255)

            hls_colorlower=(110,43,26)
            hls_colorupper=(130,127,255)

            lab_colorlower=(48,127,122)
            lab_colorupper=(92,207,27)

            bgr_colorlower=(54,46,42)
            bgr_colorupper=(255,0,85) 

        elif color == 'GREEN':
            # http://colorizer.org/ for color mask computation
            hsv_colorlower=(29, 86, 15)
            hsv_colorupper=(90, 255, 255)

            #hsv_colorlower=(25, 85, 5)
            #hsv_colorupper=(60, 255, 255)

            hls_colorlower=(25,5,51)
            hls_colorupper=(60,127,255)

            lab_colorlower=(48,127,122)
            lab_colorupper=(92,207,27)

            bgr_colorlower=(61,91,92)
            bgr_colorupper=(34,255,0)

        elif color == 'RED':
            hsv_colorlower=(0, 50, 50)
            hsv_colorupper=(10, 255, 255)
            hsv_colorlower1=(170, 50, 50)
            hsv_colorupper1=(180, 255, 255)


        """ Display Different colors """
        if debug_camera_trace >= 'DEBUG_LEVEL3':
            dbg_height = 240
            dbg_width = 320
            dbg_img = np.zeros((dbg_height,dbg_width,3), np.uint8)

            """ Build BGR image from lower and upper mask """
            dbg_img[:,0:dbg_width]=bgr_colorlower
            cv2.namedWindow('bgr_image_lower',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('bgr_image_lower',320,240)
            cv2.imshow('bgr_image_lower',dbg_img)
            cv2.waitKey(1)

            dbg_img[:,0:dbg_width]=bgr_colorupper
            cv2.namedWindow('bgr_image_upper',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('bgr_image_upper',320,240)
            cv2.imshow('bgr_image_upper',dbg_img)
            cv2.waitKey(1)

            """ Build HSV image from lower and upper mask """
            dbg_img[:,0:dbg_width]=hsv_colorlower
            hsv_img = cv2.cvtColor(dbg_img, cv2.COLOR_HSV2BGR)
            cv2.namedWindow('hsv_image_lower',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hsv_image_lower',320,240)
            cv2.imshow('hsv_image_lower',hsv_img)
            cv2.waitKey(1)

            dbg_img[:,0:dbg_width]=hsv_colorupper
            hsv_img = cv2.cvtColor(dbg_img, cv2.COLOR_HSV2BGR)
            cv2.namedWindow('hsv_image_upper',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hsv_image_upper',320,240)
            cv2.imshow('hsv_image_upper',hsv_img)
            cv2.waitKey(1)

            """ Build HLS image from lower and upper mask """
            dbg_img[:,0:dbg_width]=hls_colorlower
            hls_img = cv2.cvtColor(dbg_img, cv2.COLOR_HLS2BGR)
            cv2.namedWindow('hls_image_lower',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hls_image_lower',320,240)
            cv2.imshow('hls_image_lower',hls_img)
            cv2.waitKey(1)

            dbg_img[:,0:dbg_width]=hls_colorupper
            hls_img = cv2.cvtColor(dbg_img, cv2.COLOR_HLS2BGR)
            cv2.namedWindow('hls_image_upper',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hls_image_upper',320,240)
            cv2.imshow('hls_image_upper',hls_img)
            cv2.waitKey(1)

            """ Build BGR image from lower and upper mask """
            dbg_img[:,0:dbg_width]=lab_colorlower
            lab_img = cv2.cvtColor(dbg_img, cv2.COLOR_LAB2BGR)
            cv2.namedWindow('lab_image_lower',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('lab_image_lower',320,240)
            cv2.imshow('lab_image_lower',lab_img)
            cv2.waitKey(1)

            dbg_img[:,0:dbg_width]=lab_colorupper
            lab_img = cv2.cvtColor(dbg_img, cv2.COLOR_LAB2BGR)
            cv2.namedWindow('lab_image_upper',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('lab_image_upper',320,240)
            cv2.imshow('lab_image_upper',lab_img)
            cv2.waitKey(1)



        # resize the frame, blur it, and convert it to the HSV
        # color space
        #frame = imutils.resize(frame, width=640)
        #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ####print('lab:',lab)

        # construct a mask for the selected color,  then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask_hsv = cv2.inRange(hsv, hsv_colorlower, hsv_colorupper)
        mask_hls = cv2.inRange(hls, hls_colorlower, hls_colorupper)
        mask_lab = cv2.inRange(lab, lab_colorlower, lab_colorupper)
        if color == 'RED':
            mask1_hsv=cv2.inRange(hsv, hsv_colorlower1, hsv_colorupper1)
            mask_hsv= mask_hsv+mask1_hsv

        mask_hsv = cv2.erode(mask_hsv, None, iterations=2)
        mask_hsv = cv2.dilate(mask_hsv, None, iterations=2)

        mask_hls = cv2.erode(mask_hls, None, iterations=2)
        mask_hls = cv2.dilate(mask_hls, None, iterations=2)

        mask_lab = cv2.erode(mask_lab, None, iterations=2)
        mask_lab = cv2.dilate(mask_lab, None, iterations=2)

        if debug_camera_trace >= 'DEBUG_LEVEL1':
            cv2.imshow("MASK_HSV"+cam, mask_hsv)
            cv2.imshow("MASK_HLS"+cam, mask_hls)
            #cv2.imshow("MASK_LAB"+cam, mask_lab)

        """ Extract l s channel from hls frame"""
        l_channel = hls[:,:,1]
        l_bin = np.zeros_like(l_channel)
        l_bin[(l_channel > 30) & (l_channel <= 180)] = 255 # ref for blue
        #l_bin[(l_channel > 30) & (l_channel <= 180)] = 255 # ref for green

        s_channel = hls[:,:,2]
        s_bin = np.zeros_like(s_channel)
        s_bin[(s_channel > 100) & (s_channel <= 255)] = 255  # ref for blue
        #s_bin[(s_channel > 90) & (s_channel <= 255)] = 255

        # combine l & s channel mask
        ls_bin = np.zeros_like(l_channel)
        ls_bin[((s_bin == 255) & (l_bin == 255))] = 255
        # erode and dilate mask
        ls_bin = cv2.erode(ls_bin, None, iterations=2)
        ls_bin = cv2.dilate(ls_bin, None, iterations=2)

        if debug_camera_trace >= 'DEBUG_LEVEL2':
            cv2.namedWindow('l_hls_bin',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('l_hls_bin',320,240)
            cv2.imshow('l_hls_bin',l_bin)
            cv2.waitKey(1)

            cv2.namedWindow('s_hls_bin',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('s_hls_bin',320,240)
            cv2.imshow('s_hls_bin',s_bin)
            cv2.waitKey(1)

            cv2.namedWindow('ls_hls_bin',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ls_hls_bin',320,240)
            cv2.imshow('ls_hls_bin',ls_bin)
            cv2.waitKey(1)

        """ Calc magnitude of gradient binary mask """
        mag_bin = mag_thresh(frame, sobel_kernel=3, mag_thresh=(50, 255))
        mag_bin = mag_bin * 255
        if debug_camera_trace >= 'DEBUG_LEVEL2':
            cv2.namedWindow('mag_thresh',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('mag_thresh',320,240)
            cv2.imshow('mag_thresh',mag_bin)



        """ Part where different mask are combined together """
        mask_comb = np.zeros_like(s_bin)
        #mask_comb[(ls_bin == 255)]=255
        #mask_comb[(mask_hsv == 255)]=255 # Reference for green
        mask_comb[((mask_hsv == 255) & (ls_bin == 255))] = 255 # Reference for blue
        #mask_comb[((mask_hsv == 255) | (mask_hls == 255))] = 255
        #mask_comb[((mask_hsv == 255) & (mask_lab == 255))] = 255
        if debug_camera_trace >= 'DEBUG_LEVEL1':
            cv2.imshow("MASK COMBINED"+cam, mask_comb)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask_comb.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] !=0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (int(x), int(y))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        if pts is not None:
            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in xrange(1, len(pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[i - 1] is None or pts[i] is None:
                            continue

                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the frame to our screen
        cv2.imshow(cam, frame)







# Class vehicule
class Vehicule:
    
    
    def __init__(self, kwargs):
        valid_args = set(['name','color1','color2','xpos','ypos'])
        self.__dict__.update((arg, None) for arg in valid_args)
        self.__dict__.update((arg, val) for arg, val in kwargs.items() if arg in valid_args)
        self.c1 = []
        self.c2 = []
        # Needed, as None value will break future format converting
        self.xpos = 0
        self.ypos = 0
        
        self.gpsOrient = 0.0
        
        self.gpsSpeed=0
        self.firstGpsPostionReceived = False
        self.errorDistance = 0
        self.lastGpsPositionTime = time.time()
        self.lastGpsPositionn = (0,0)
        self.lastGpsPositionCheck = (0,0)
        self.lastGpsPositionTimeCheck = time.time()
    
    def getxpos(self):
        return self.xpos

    def getypos(self):
        return self.ypos
    
    def getSpeed(self):
        return self.gpsSpeed

    def getOrient(self):
        return self.gpsOrient
    
    def getname(self):
        return self.name

    def color_mask(self, c, frame):
        if len(c) == 4:
            # case 4 color boundaries
            lower1 = np.array(c[0])
            upper1 = np.array(c[1])
            lower2 = np.array(c[2])
            upper2 = np.array(c[3])
            mask1 = cv2.inRange(frame, lower1, upper1) 
            mask2 = cv2.inRange(frame, lower2, upper2)
            return (mask1|mask2)
        
        elif len(c) == 2:
            # case 2 color boundaries
            lower = np.array(c[0])
            upper = np.array(c[1])
            return cv2.inRange(frame, lower, upper)

    def calc_pos(self, frame, ref_frame, ratio):
        self.xpos = -1
        self.ypos = -1
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # TODO POS on color 1 ONLY
        mask_hsv = self.color_mask(self.c1, frame_hsv)
        # Add more processing - erode / dilate / 2nd GaussianBlur
        mask_hsv = cv2.erode(mask_hsv, None, iterations=2)
        mask = cv2.dilate(mask_hsv, None, iterations=2)
        #mask = cv2.GaussianBlur(mask, (9, 9), 2)
        # cv2.imshow("CAM MASK", mask_hsv)
        # Find contours in the frame
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # only proceed if the radius meets a minimum size
            if radius > 5:
                self.xpos = int(x*ratio)
                self.ypos = int(y*ratio)   
                    
                #print "FIX :", self.xpos, self.ypos
                #cv2.circle(ref_frame, (self.xpos, self.ypos), int(radius*ratio), (0, 255, 255), 2)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(ref_frame, self.name, (self.xpos, self.ypos - 20), font, 0.5, (0, 255, 255),2,cv2.CV_AA)
                
                #cv2.putText(ref_frame, self.name, (self.xpos, self.ypos - 20), font, 0.5, (0, 255, 255),2,cv2.LINE_AA)
                # M = cv2.moments(c)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                # cv2.circle(frame, center, 5, (0, 0, 255), -1)

    def calc_speedOrient(self,x,y):
        #record the first position of the car for the speed 
        if self.firstGpsPostionReceived == False:
            self.firstGpsPostionReceived = True
            #init last pos
            self.lastGpsPositionTime = time.time()
            self.lastGpsPosition = (x,y)
            self.gpsSpeed = 0
        else:
            #compute orientation compared to last point 
            self.gpsOrient = round(atan2((y-self.lastGpsPosition[1]), (x-self.lastGpsPosition[0])),3)
            
            timeNow = time.time()
            #compute speed about every half second
            if timeNow > (self.lastGpsPositionTime + SPEED_SAMPLING_TIME):
                #compute new distance
                dist = distance((x,y),self.lastGpsPosition)
                
                #compute new speed in mm/s
                self.gpsSpeed = dist / (timeNow - self.lastGpsPositionTime)
                #send only when gps step is significant
                #update last pos
                self.lastGpsPositionTime = time.time()
                self.lastGpsPosition = (x,y)
                
            
    def check_move(self,x,y):
        #record the first position of the car for the speed
        dist = distance((x,y),self.lastGpsPositionCheck)
        timeNow = time.time()
        if dist > MAX_CAR_SPEED*(timeNow - self.lastGpsPositionTimeCheck):
            print 'Move impossible for car ',self.name
            res = False
        else :
            res = True
            
        self.lastGpsPositionTimeCheck = timeNow
        self.lastGpsPositionCheck = (x,y)            
        return res
    
                

class gpsFixThread(VideoThread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(gpsFixThread, self).__init__()
        print('Gps Camera Calib file:', cam_gps_calib_file)
        print('Circuit Zone of Interrest file:', circuit_zone_of_interest_file)

        

        
 
    # used to receive jpg image in client thread 
    def _handle_RECEIVE(self, cmd):
        global mouseXY
        #initialize th evehicule list according to config
        for data in VEHICULE_LIST:
            list_of_v.append(Vehicule(data))
        for v in list_of_v:
            v.c1 = COLOR[v.color1]
            if v.color2 in COLOR.keys():
                v.c2 = COLOR[v.color2]
                
        lastGpsFixTime=time.time()   
        
        with open(cam_gps_calib_file, 'rb') as f:
            save_dict = pickle.load(f)
        mtx = save_dict['mtx']
        dist = save_dict['dist']
        print ('mtx:', mtx)
        print ('dist:', dist)
    
        with open(circuit_zone_of_interest_file, 'rb') as f:
            save_dict_circuit = pickle.load(f)
        circuit_roi = save_dict_circuit['roi']
        user_roi = save_dict_circuit['user_roi']
        scale_XY = save_dict_circuit['scale']
        print('user_roi:', user_roi)
        print('circuit roi:', circuit_roi)
        print('scale XY:', scale_XY)
        
        alpha = 0.05
        
        gpsSpeed = 0
        
        imageOrigine = None

        #set mouse callback
        cv2.namedWindow("GPS detect area", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GPS detect area", 640, 480)
        cv2.setMouseCallback("GPS detect area", mousePosition);
        
        if self.connected == True:
            while True:
                #check first if new command to stop comes in
                try:
                    newCmd = self.cmd_q.get(False)
                    if newCmd.type == ClientCommand.STOP:
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
                        ImageJpg = self.rcvBytes[a:b+2]

                        #now shift rcvbyte to manage next image
                        self.rcvBytes=self.rcvBytes[b+2:]
                        
                        imageOrigine = cv2.imdecode(np.fromstring(ImageJpg, dtype=np.uint8),-1)
                        
                        if showAllImage == True :
                            cv2.imshow("Original Frame", imageOrigine)
                            cv2.waitKey(1)


                    #handle new image only if it is time to do it
                    timeNow = time.time()
                    if timeNow > (lastGpsFixTime + GPS_FIX_BROADCAST_DELAY) and imageOrigine != None:

                        # refine the camera Matrix / Alpha can be be changed from 0 to 1 in our case 0.1 is enough
                        h,  w = imageOrigine.shape[:2]
                        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
                
                        frame_undist = cv2.undistort(imageOrigine, mtx, dist, None, newcameramtx)
                
                        # crop the image
                        x, y, w, h = roi
                        frame_undist = frame_undist[y:y+h, x:x+w]
                        
                        
                         # Crop image per Rectangle ROI calibration
                        # Note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                        #crop_frame = frame_undist[circuit_roi[1]:circuit_roi[5], circuit_roi[0]:circuit_roi[2]] # Crop from x, y, w, h
                        crop_frame = frame_undist[user_roi[1]:user_roi[5], user_roi[0]:user_roi[2]] # Crop from x, y, w, h
                        
                        ref_frame = crop_frame
                        #ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)

                        if RESIZE > 5:
                            frame = imutils.resize(ref_frame, width=RESIZE)
                            ratio = ref_frame.shape[0] / float(frame.shape[0])
                        else:
                            frame = ref_frame
                            ratio = 1

                        # Image optimization, normalization
                        # height, width, channels = frame.shape
                        # rgb = np.zeros((height, width, channels),np.float32)
                        # comb = np.zeros((height, width, channels),np.float32)
                        # rgb[:,:,:] = frame[:,:,:]
                        # B_channel = rgb[:,:,0]
                        # G_channel = rgb[:,:,1]
                        # R_channel = rgb[:,:,2]
                        # sum = B_channel + G_channel + R_channel
                        # comb[:,:,0] = B_channel/sum*255.0
                        # comb[:,:,1] = G_channel/sum*255.0
                        # comb[:,:,2] = R_channel/sum*255.0
                        # frame = cv2.convertScaleAbs(comb)
                        # cv2.imshow("RGB view", ref_frame)
                        # Image optimization, first stage medianBlur + HSV color space
                        # frame = cv2.medianBlur(frame, 3)
                        # TODO Test HLS format
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        # Identify vehicule color position on frame
                        for v in list_of_v:
                            v.calc_pos(frame, ref_frame, ratio)
                            x=v.getxpos()
                            y=v.getypos()

                            #check if we don't have negative value that is not expected for gps
                            if (x < 0) :
                                print 'Error : Negative point for ',v.getname()
                                continue
                            

                            #convert into mm
                            x_mm=int(x*scale_XY[0])
                            y_mm=int(y*scale_XY[1])
                            
                            if v.check_move(x_mm,y_mm) == True:
                                v.calc_speedOrient(x_mm,y_mm)
                                speed = int(v.getSpeed())
                                orient = v.getOrient()

                                v_data = str(int(x_mm)) + ',' + str(int(y_mm)) + ',' + str(int(speed)) + ',' +str(orient) + ',' + v.getname()
                                self.reply_q.put(self._success_reply(v_data))
                        
                        
                        # compute hsv or svh in ordr to print out the color range on the mouse
                        
                        hsv = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2HSV)
                        
                        #display out few value
                        for v in list_of_v:
                            x=v.getxpos()
                            y=v.getypos()

                            #to be replace when we want to move in mm
                            x_mm=int(x*scale_XY[0])
                            y_mm=int(y*scale_XY[1])

                            #check if we don't have negative value that is not expected for gps
                            if (x < 0) :
                                continue   
                                                         
                            speed=int(v.getSpeed())
                                                            
                            radius=2
                            cv2.circle(ref_frame, (x, y), int(radius*ratio), (0, 255, 255), 2)
                            cv2.putText(ref_frame,"x="+str(x_mm)+" y="+str(y_mm)+" v="+str(speed),(x+4, y),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        h = hsv[mouseXY[1], mouseXY[0]][0];
                        s = hsv[mouseXY[1], mouseXY[0]][0];
                        v = hsv[mouseXY[1], mouseXY[0]][0];

                        cv2.putText(crop_frame,'XYpixel  = ' + str(mouseXY),
                                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(crop_frame,'XY(mm) = ' + '(' + str(round(mouseXY[0] * scale_XY[0], 2)) + ', ' + str(round(mouseXY[1] * scale_XY[1], 2)) + ')',
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(ref_frame,'HSV color XY = ' + str(hsv[mouseXY[1], mouseXY[0]]),
                                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        
                        
                        
                                            
                        cv2.imshow("GPS detect area", ref_frame)
                        cv2.waitKey(1)
                                      
                                    
                        # Debug 
                        #cv2.imshow("CAM view", ref_frame)  

                        lastGpsFixTime = timeNow

                except IOError as e:
                    self.reply_q.put(self._error_reply(str(e)))


        

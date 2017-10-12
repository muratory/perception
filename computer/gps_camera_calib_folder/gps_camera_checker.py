# import the necessary packages
import numpy as np
import argparse
import time
import io
import cv2
import urllib
import socket
import pickle

from imutils.video import VideoStream
from matplotlib import pyplot as plt

#GPS_IP = '192.168.0.147'
#GPS_IP = '192.168.0.54'
GPS_IP = '192.168.0.81' # GPS IP RASPBERRY GPS CAM
PORT_VIDEO_SERVER = 8020

WIN_WIDTH = 1024
WIN_HEIGHT = 768

mouseXY = (0,0)

def parseArguments():
    """
    Parse the command line.
    @return a list of arguments
    """
    parser = argparse.ArgumentParser(
        description="Check All Calibration parameter for GPS camera.", )

    parser.add_argument("-c", "--calib", dest="calib", required=True,
                        help="gps camera calibration parameters file")

    parser.add_argument("-r", "--roi", dest="roi", required=True,
                        help="circuit zone of interrest parameters file")

    parser.add_argument("-a", "--alpha", dest="alpha", default = 0.05, required=False,
                        help="choose alpha value from range [0. : 1.] for Optimal Camerai Matrix")

    parser.add_argument("-i", "--image", dest="image", required=False,
                        help="checker will use image as input instead of video flux")

    return parser.parse_args()

def mousePosition(event,x,y,flags,param):

    if event == cv2.EVENT_MOUSEMOVE:
        global mouseXY
        mouseXY=(x,y)

def frameProcessing(frame, frame_nb, mtx, dist, alpha, circuit_roi, user_roi, scale_XY):

    # refine the camera Matrix / Alpha can be be changed from 0 to 1 in our case 0.1 is enough
    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))

    frame_undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    frame_undist = frame_undist[y:y+h, x:x+w]
    #cv2.imshow("Undist Frame", frame_undist)
    cv2.rectangle(frame_undist, (circuit_roi[0], circuit_roi[1]), (circuit_roi[4], circuit_roi[5]), (148, 236, 178), 1, cv2.LINE_AA)
    cv2.arrowedLine(frame_undist, (circuit_roi[0], (circuit_roi[5] - circuit_roi[1]) // 2 + circuit_roi[1]),
                                  (circuit_roi[4], (circuit_roi[5] - circuit_roi[1]) // 2 + circuit_roi[1]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(frame_undist, (circuit_roi[4], (circuit_roi[5] - circuit_roi[1]) // 2 + circuit_roi[1]),
                                  (circuit_roi[0], (circuit_roi[5] - circuit_roi[1]) // 2 + circuit_roi[1]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(frame_undist, ((circuit_roi[4] - circuit_roi[0])*1//4 + circuit_roi[0], circuit_roi[1]),
                                  ((circuit_roi[4] - circuit_roi[0])*1//4 + circuit_roi[0], circuit_roi[5]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(frame_undist, ((circuit_roi[4] - circuit_roi[0])*1//4 + circuit_roi[0], circuit_roi[5]),
                                  ((circuit_roi[4] - circuit_roi[0])*1//4 + circuit_roi[0], circuit_roi[1]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(frame_undist, ((circuit_roi[4] - circuit_roi[0])*3//4 + circuit_roi[0], circuit_roi[1]),
                                  ((circuit_roi[4] - circuit_roi[0])*3//4 + circuit_roi[0], circuit_roi[5]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(frame_undist, ((circuit_roi[4] - circuit_roi[0])*3//4 + circuit_roi[0], circuit_roi[5]),
                                  ((circuit_roi[4] - circuit_roi[0])*3//4 + circuit_roi[0], circuit_roi[1]), (148, 236, 178), 1, cv2.LINE_AA, 0, 0.025)

    cv2.putText(frame_undist, 'LC=' + str(round((circuit_roi[4] - circuit_roi[0]) * scale_XY[0]/10, 2)) + ' cm',
                ( (circuit_roi[4] - circuit_roi[0]) * 1//8 , (circuit_roi[5] - circuit_roi[1])//2 + circuit_roi[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 236, 178), 1, cv2.LINE_AA)
    cv2.putText(frame_undist, 'HC=' + str(round((circuit_roi[5] - circuit_roi[1]) * scale_XY[1]/10, 2)) + ' cm',
                ( (circuit_roi[4] - circuit_roi[0]) //4 + circuit_roi[0] + 2, (circuit_roi[5] - circuit_roi[1]) * 5 // 8 + circuit_roi[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 236, 178), 1, cv2.LINE_AA)

    # Crop image per Rectangle ROI calibration
    # Note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    #crop_frame = frame_undist[circuit_roi[1]:circuit_roi[5], circuit_roi[0]:circuit_roi[2]] # Crop from x, y, w, h
    crop_frame = frame_undist[user_roi[1]:user_roi[5], user_roi[0]:user_roi[2]] # Crop from x, y, w, h

    cv2.line(crop_frame, (0, 0), (crop_frame.shape[1] - 1, crop_frame.shape[0] - 1), (0, 0, 255), 1)
    cv2.line(crop_frame, (0, crop_frame.shape[0]), (crop_frame.shape[1] - 1, 0), (0, 0, 255), 1)
    #cv2.line(crop_frame, (0, crop_frame.shape[0]//2), (crop_frame.shape[1] - 1, crop_frame.shape[0]//2), (255, 0, 0), 1)
    cv2.arrowedLine(crop_frame, (0, crop_frame.shape[0]//2), (crop_frame.shape[1] - 1, crop_frame.shape[0]//2), (255, 0, 0), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(crop_frame, (crop_frame.shape[1] - 1, crop_frame.shape[0]//2), (0, crop_frame.shape[0]//2), (255, 0, 0), 1, cv2.LINE_AA, 0, 0.025)
    cv2.line(crop_frame, ((crop_frame.shape[1] - 1)*1//4, 0), ((crop_frame.shape[1] - 1)*1//4, crop_frame.shape[0]-1), (255, 0, 0), 1)
    #cv2.line(crop_frame, ((crop_frame.shape[1] - 1)*3//4, 0), ((crop_frame.shape[1] - 1)*3//4, crop_frame.shape[0]-1), (255, 0, 0), 1)
    cv2.arrowedLine(crop_frame, ((crop_frame.shape[1] - 1)*3//4, 0),
                                ((crop_frame.shape[1] - 1)*3//4, crop_frame.shape[0]-1), (255, 0, 0), 1, cv2.LINE_AA, 0, 0.025)
    cv2.arrowedLine(crop_frame, ((crop_frame.shape[1] - 1)*3//4, crop_frame.shape[0]-1),
                                ((crop_frame.shape[1] - 1)*3//4, 0), (255, 0, 0), 1, cv2.LINE_AA, 0, 0.025)


    cv2.putText(crop_frame, 'Width:  1px = ' + str(round(scale_XY[0], 2)) + ' mm',
                (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(crop_frame, 'Height: 1px = ' + str(round(scale_XY[1], 2)) + ' mm',
                (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(crop_frame, 'L=' + str(round(crop_frame.shape[1]*scale_XY[0]/10, 2)) + ' cm',
                (crop_frame.shape[1] * 1//8, crop_frame.shape[0]//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(crop_frame, 'H=' + str(round(crop_frame.shape[0]*scale_XY[1]/10, 2)) + ' cm',
                (crop_frame.shape[1] * 1//4, crop_frame.shape[0] * 5//8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(crop_frame, 'Frame=' + str(frame_nb),
                (1, crop_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    #cv2.putText(crop_frame,'XYpixel = ' + str(mouseXY),
                #(crop_frame.shape[1] * 1//4, crop_frame.shape[0] * 1//4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    #cv2.putText(crop_frame,'XY(mm) = ' + '(' + str(round(mouseXY[0] * scale_XY[0], 2)) + ', ' + str(round(mouseXY[1] * scale_XY[1], 2)) + ')',
                #(crop_frame.shape[1] * 1//4, crop_frame.shape[0] * 1//4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(crop_frame,'XYpixel  = ' + str(mouseXY),
                (crop_frame.shape[1] * 1//4 + 10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(crop_frame,'XY(mm) = ' + '(' + str(round(mouseXY[0] * scale_XY[0], 2)) + ', ' + str(round(mouseXY[1] * scale_XY[1], 2)) + ')',
                (crop_frame.shape[1] * 1//4 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
    h = hsv[mouseXY[1], mouseXY[0]][0];
    s = hsv[mouseXY[1], mouseXY[0]][0];
    v = hsv[mouseXY[1], mouseXY[0]][0];

    cv2.putText(crop_frame,'HSV color XY = ' + str(hsv[mouseXY[1], mouseXY[0]]),
                (crop_frame.shape[1] * 1//4 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


    return crop_frame


if __name__ == "__main__":

    args = parseArguments()
    cam_gps_calib_file = args.calib
    circuit_zone_of_interest_file = args.roi
    alpha = float(args.alpha)
    print('args:', args) 
    print('Gps Camera Calib file:', cam_gps_calib_file)
    print('Circuit Zone of Interrest file:', circuit_zone_of_interest_file)


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

    frame_nb = 0

    # open the webcam
    #cam = cv2.VideoCapture(0)
    #if ( not cam.isOpened() ):
         #print "no cam!"
         #sys.exit()
    #print "cam: ok."



    #cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1024);

    """ 
    0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    3. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
    4. cV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    5. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    6. CV_CAP_PROP_FPS Frame rate.
    7. CV_CAP_PROP_FOURCC 4-character code of codec.
    8. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    9. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    10. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    11. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    12. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    13. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    14. CV_CAP_PROP_HUE Hue of the image (only for cameras).
    15. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    16. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    17. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    18. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
    19. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cam (supported by DC1394 v 2.x backend currently) 
    """
    #print ('cv2.CAP_PROP_POS_MSEC:',cam.get(cv2.CAP_PROP_POS_MSEC))
    #print ('cv2.CAP_PROP_POS_FRAMES:',cam.get(cv2.CAP_PROP_POS_FRAMES))
    #print ('cv2.CAP_PROP_POS_AVI_RATIO:',cam.get(cv2.CAP_PROP_POS_AVI_RATIO))
    #print ('cv2.CAP_PROP_FRAME_WIDTH:',cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    #print ('cv2.CAP_PROP_FRAME_HEIGHT:',cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print ('cv2.CAP_PROP_FPS:',cam.get(cv2.CAP_PROP_FPS))
    #print ('cv2.CAP_PROP_FOURCC:',cam.get(cv2.CAP_PROP_FOURCC))
    #print ('cv2.CAP_PROP_FRAME_COUNT:',cam.get(cv2.CAP_PROP_FRAME_COUNT))
    #print ('cv2.CAP_PROP_FORMAT:',cam.get(cv2.CAP_PROP_FORMAT))
    #print ('cv2.CAP_PROP_MODE:',cam.get(cv2.CAP_PROP_MODE))
    #print ('cv2.CAP_PROP_BRIGHTNESS:',cam.get(cv2.CAP_PROP_BRIGHTNESS))
    #print ('cv2.CAP_PROP_CONTRAST:',cam.get(cv2.CAP_PROP_CONTRAST))
    #print ('cv2.CAP_PROP_SATURATION:',cam.get(cv2.CAP_PROP_SATURATION))
    #print ('cv2.CAP_PROP_HUE:',cam.get(cv2.CAP_PROP_HUE))
    #print ('cv2.CAP_PROP_GAIN:',cam.get(cv2.CAP_PROP_GAIN))
    #print ('cv2.CAP_PROP_EXPOSURE:',cam.get(cv2.CAP_PROP_EXPOSURE))
    #print ('cv2.CAP_PROP_CONVERT_RGB:',cam.get(cv2.CAP_PROP_CONVERT_RGB))
    ####print ('cv2.CAP_PROP_WHITE_BALANCE:',cam.get(cv2.CAP_PROP_WHITE_BALANCE))
    #print ('cv2.CAP_PROP_RECTIFICATION:',cam.get(cv2.CAP_PROP_RECTIFICATION))



    if args.image is not None:
        print ('Picture Mode:', args.image) 
    else:
        s1 = 'http://' + GPS_IP + ':' + str(PORT_VIDEO_SERVER)
        print ('Video Mode:', s1)
        rcvBytes = ''
        try:
            stream = urllib.urlopen('http://' + GPS_IP + ':' + str(PORT_VIDEO_SERVER) + '/?action=stream')
        except IOError as e:
            print ('Error: Fail to open url stream / Check streamer on raspberry board IP/PORT ...')
            pass


    
    cv2.namedWindow('Cropped Frame:', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cropped Frame:', WIN_WIDTH, WIN_HEIGHT)
    cv2.moveWindow('Cropped Frame:', 0 , 0)

    cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Frame', WIN_WIDTH, WIN_HEIGHT)

    #set mouse callback
    cv2.setMouseCallback('Cropped Frame:', mousePosition);
    

    while True:
        frame_nb = frame_nb + 1
        if args.image is not None:
            frame = cv2.imread(args.image)
        else:
            #loop until image found or problem
            rcvBytes += stream.read(1024)
            #print 'rcv = ' + str(len(rcvBytes))
            # search for jpg image
            a = rcvBytes.find('\xff\xd8')
            b = rcvBytes.find('\xff\xd9')
            #print('n')
            if a!=-1 and b!=-1:
                #image found , send it in receive queue
                frame_jpg = rcvBytes[a:b+2]
                #now shift rcvbyte to manage next image
                #print ('len(rcvBytes):', len(rcvBytes) , len(frame_jpg) )
                rcvBytes=rcvBytes[b+2:]
                #print ('len(rcvBytes):', len(rcvBytes) )
                frame = cv2.imdecode(np.fromstring(frame_jpg, dtype=np.uint8),-1)
            else:
                continue

        cv2.imshow("Original Frame", frame)

        # Process the frame
        crop_frame = frameProcessing(frame, frame_nb, mtx, dist, alpha, circuit_roi, user_roi, scale_XY)

        cv2.imshow("Cropped Frame:", crop_frame)
        

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break
            
        


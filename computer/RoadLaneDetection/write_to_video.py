from __future__ import print_function

import argparse
import cv2

from os import listdir
from os.path import isfile, join

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to output video file")
ap.add_argument("-i", "--image", required =True,
    help = "Path to the image folder")
ap.add_argument("-f", "--fps", type=int, default=20,
    help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
    help="codec of output video")
args = vars(ap.parse_args())


mypath = args["image"]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
 
# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None

# loop and look for the min (w,h) before to encode 
for n in range(0, len(onlyfiles)):
    frame = cv2.imread(join(mypath, onlyfiles[n]))
    if (h, w) != (None, None):
        if frame.shape[0] < h:
            (h, w) = (frame.shape[0], w)
        if frame.shape[1] < w:
            (h, w) = (h, frame.shape[1])
    else:
        (h, w) = frame.shape[:2]

print ('(h, w):', (h, w))

# loop over frames from the video stream
for n in range(0, len(onlyfiles)):
    # grab the frame from the video stream and resize it to have a
    # maximum width of 300 pixels
    frame = cv2.imread(join(mypath, onlyfiles[n]))
    print ("frame.shape:", frame.shape)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    dim = (w, h)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # print ("frame:", frame.shape)
 
    # check if the writer is None
    if writer is None:
        # store the image dimensions, initialize the video writer,
        # and construct the zeros array
        (h, w) = frame.shape[:2]
        print ("(h, w):", h, w)
        writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h), True)

    # write the output frame to file
    writer.write(frame)

    # show the frames
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
writer.release()
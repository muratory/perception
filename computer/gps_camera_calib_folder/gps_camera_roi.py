#!/usr/bin/env python
# -*- coding: utf-8 -

"""
"""

# ----------------------  IMPORTS  -------------------------
import argparse

import cv2
import numpy as np
import pickle

from basics import Line

WIN_WIDTH = 1024
WIN_HEIGHT = 768
CIRCUIT_WIDTH = 4710.  # Unit milimeter
CIRCUIT_HEIGHT = 2325. # Unit milimeter

def parseArguments():
    """
    Parse the command line.
    @return a list of arguments
    """
    parser = argparse.ArgumentParser(
        description="Defines a region of interest.", )

    parser.add_argument("-i", "--image", dest="image", required=True,
                        help="input source path")

    return parser.parse_args()


def nothing(position):
    # print position
    pass

def setXYBounds(image):
    """
    Define a strip of interest by retrieving the Y bounds.
    :param image: image to process
    :return: a tuple with Y bounds
    """
    height, width = image.shape[:2]
    yMin, yMax = height * 5 // 100, height * 95 // 100
    xMin, xMax = width * 5 // 100, width * 95 // 100

    cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI', WIN_WIDTH, WIN_HEIGHT)
    cv2.createTrackbar('Top', 'ROI', yMin, height, nothing)
    cv2.createTrackbar('Bottom', 'ROI', yMax, height, nothing)
    cv2.createTrackbar('Left', 'ROI', xMin, width, nothing)
    cv2.createTrackbar('Right', 'ROI', xMax, width, nothing)


    while (1):
        output = image.copy()

        yMin = cv2.getTrackbarPos('Top', 'ROI')
        yMax = cv2.getTrackbarPos('Bottom', 'ROI')
        xMin = cv2.getTrackbarPos('Left', 'ROI')
        xMax = cv2.getTrackbarPos('Right', 'ROI')

        cv2.line(output, (0, yMin), (width - 1, yMin), (0, 255, 255), 1)
        cv2.line(output, (0, yMax), (width - 1, yMax), (0, 255, 255), 1)
        cv2.line(output, (xMin, 0), (xMin, height - 1), (0, 255, 255), 1)
        cv2.line(output, (xMax, 0), (xMax, height - 1), (0, 255, 255), 1)
        cv2.line(output, (xMin, yMin), (xMax, yMax), (0, 0, 255), 1)
        cv2.line(output, (xMin, yMax), (xMax, yMin), (0, 0, 255), 1)

        cv2.imshow('ROI', output)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return xMin, xMax, yMin, yMax


def setXYBoundsForCropping(image,roi):
    """
    Define a strip of interest by retrieving the Y bounds.
    :param image: image to process
    :return: a tuple with Y bounds
    """

    # Retreive the first zone coord
    xMin = roi[0]
    xMax = roi[2]
    yMin = roi[1]
    yMax = roi[5]

    # Determine the (x0,y0) for center
    x0 = (xMax - xMin) // 2
    y0 = (yMax - yMin) // 2

    # Compute and determine equation for diagonale 
    lines_x = []
    lines_y = []
    lines_x.append(xMin)
    lines_x.append(xMax)
    lines_y.append(yMin)
    lines_y.append(yMax)

    lines_x1 = []
    lines_y1 = []
    lines_x1.append(xMin)
    lines_x1.append(xMax)
    lines_y1.append(yMax)
    lines_y1.append(yMin)


    m, b = np.polyfit(lines_x, lines_y, 1)  # y = m*x + b
    m1, b1 = np.polyfit(lines_x1, lines_y1, 1) # y = m1*x + b1
    
    XMin = xMin
    YMin = m*XMin + b

    YMax = m1 * XMin + b1
    XMax = (YMax - b) / m

    pixel_width_scale = CIRCUIT_WIDTH/(xMax - xMin)

    cv2.namedWindow('CROP ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CROP ROI', WIN_WIDTH, WIN_HEIGHT)
    cv2.createTrackbar('ratio', 'CROP ROI', XMin, x0+xMin, nothing)


    while (1):
        output = image.copy()

        XMin = cv2.getTrackbarPos('ratio', 'CROP ROI')
        YMin = m*XMin + b

        YMax = m1 * XMin + b1
        XMax = (YMax - b) / m

        cv2.line(output, (xMin, yMin), (xMax, yMin), (0, 255, 255), 1)
        cv2.line(output, (xMax, yMin), (xMax, yMax), (0, 255, 255), 1)
        cv2.line(output, (xMax, yMax), (xMin, yMax), (0, 255, 255), 1)
        cv2.line(output, (xMin, yMax), (xMin, yMin), (0, 255, 255), 1)
        cv2.line(output, (xMin, yMin), (xMax, yMax), (0, 0, 255), 1)
        cv2.line(output, (xMin, yMax), (xMax, yMin), (0, 0, 255), 1)

        cv2.line(output, (int(XMin), int(YMin)), (xMax, yMax), (255, 0, 0), 1)

        cv2.line(output, (int(XMin), int(YMin)), (int(XMax), int(YMin)), (0, 255, 0), 1)
        cv2.line(output, (int(XMax), int(YMin)), (int(XMax), int(YMax)), (0, 255, 0), 1)
        cv2.line(output, (int(XMax), int(YMax)), (int(XMin), int(YMax)), (0, 255, 0), 1)
        cv2.line(output, (int(XMin), int(YMax)), (int(XMin), int(YMin)), (0, 255, 0), 1)
        if (XMin != xMin):
            #cv2.line(output, (int(XMin), int(y0)), (int(xMin), int(y0)), (255, 0, 0), 1)
            cv2.arrowedLine(output, (int(xMin), int(y0)), (int(XMin), int(y0)), (255, 0, 0), 1, cv2.LINE_AA, 0, 0.25)
        output = cv2.putText(output, 'd= ' + str(round((xMin - XMin) * pixel_width_scale / 10, 2)) + ' cm',
                             (int(xMin), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)


        cv2.imshow('CROP ROI', output)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return int(XMin), int(XMax), int(YMin), int(YMax)


def detectROI():
    """
    main function
    :return:
    """
    args = parseArguments()
    path = args.image
    image = cv2.imread(path)
    circuit_roi = None
    user_roi = None
    print 'Press Esc key to escape/continue...'
    print 'Image: %s %s' % (path, str(image.shape))

    # Define the Rectangle 
    xyBounds = setXYBounds(image)
    if not xyBounds:
        raise Exception('Fail to define XY bounds')

    circuit_roi = (xyBounds[0], xyBounds[2],
                   xyBounds[1], xyBounds[2],
                   xyBounds[1], xyBounds[3],
                   xyBounds[0], xyBounds[3])

    # Cropping Rextangle area
    xyCroppedBounds = setXYBoundsForCropping(image, circuit_roi)
    if not xyCroppedBounds:
        raise Exception('Fail to define XY bounds For Cropping Area')
   
    user_roi = (xyCroppedBounds[0], xyCroppedBounds[2],
                xyCroppedBounds[1], xyCroppedBounds[2],
                xyCroppedBounds[1], xyCroppedBounds[3],
                xyCroppedBounds[0], xyCroppedBounds[3])
 

    # Display Undistorded Image
    cv2.namedWindow('Undist Image:', cv2.WINDOW_NORMAL)
    cv2.imshow('Undist Image:', image)

    img1 = image.copy()
    crop_img = img1[xyCroppedBounds[2]:xyCroppedBounds[3], xyCroppedBounds[0]:xyCroppedBounds[1]] # Crop from x, y, w, h 
    # Note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    img2 = image.copy()
    circuit_img = img2[xyBounds[2]:xyBounds[3], xyBounds[0]:xyBounds[1]] # Crop from x, y, w, h 
    # Note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    # Display Cropped Image
    cv2.line(crop_img, (0, 0), (crop_img.shape[1] - 1, crop_img.shape[0] - 1), (255, 0, 0), 1)
    cv2.line(crop_img, (0, crop_img.shape[0]), (crop_img.shape[1] - 1, 0), (255, 0, 0), 1)
    # Display Cropped Circuit Image
    cv2.line(circuit_img, (0, 0), (circuit_img.shape[1] - 1, circuit_img.shape[0] - 1), (255, 0, 0), 1)
    cv2.line(circuit_img, (0, circuit_img.shape[0]), (circuit_img.shape[1] - 1, 0), (255, 0, 0), 1)

    pixel_width_scale = CIRCUIT_WIDTH/circuit_img.shape[1]
    pixel_height_scale = CIRCUIT_HEIGHT/circuit_img.shape[0]
    scale_XY = (pixel_width_scale, pixel_height_scale)
    print ('Width:', circuit_img.shape[1], 'PX WIDTH SCALE = ', pixel_width_scale, 'Height:', circuit_img.shape[0], 'PX HEIGHT SCALE = ', pixel_height_scale)

    cv2.putText(crop_img, 'Width  Axis: 1px = ' + str(round(pixel_width_scale, 2)) + ' mm', (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(crop_img, 'Height Axis: 1px = ' + str(round(pixel_height_scale, 2)) + ' mm', (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.namedWindow('Circuit Image:', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Circuit Image:', WIN_WIDTH, WIN_HEIGHT)
    cv2.imshow('Circuit Image:', circuit_img)

    cv2.namedWindow('Cropped Image:', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cropped Image:', WIN_WIDTH, WIN_HEIGHT)
    cv2.imshow('Cropped Image:', crop_img)


    key = cv2.waitKey(0) & 0xFF
     
    # Create Dictionary for Rectangle ROI
    save_dict = {'roi': circuit_roi, 'user_roi': user_roi, 'scale': scale_XY}

    # Save the rectangle coord into a file (serialization)
    pos = path.rfind('.')
    if pos != -1:
        rectangle_roi_file = path[:pos] + '.tpz'
        with open(rectangle_roi_file, 'wb') as h:
            pickle.dump(save_dict, h, protocol=2)

    cv2.destroyAllWindows()


# ----------------------  MAIN  ----------------------------
if __name__ == '__main__':
    detectROI()

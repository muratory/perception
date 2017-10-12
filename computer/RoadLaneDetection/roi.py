#!/usr/bin/env python
# -*- coding: utf-8 -

"""
-------------------------------------------------------------------------------

    Module: $Title$

    Description:
        enter_a_module_description.
-------------------------------------------------------------------------------

    Copyright (c) 2017 , Intel Corporation All Rights Reserved. 


History:
Author          Date         CR           Description of Changes
-------------------------------------------------------------------------------
gcosquer      4/2/2017      bug     Creation
-------------------------------------------------------------------------------
"""

# ----------------------  IMPORTS  -------------------------
import argparse

import cv2
import numpy as np
import pickle
import global_vars as g

from processings import LineSegmentDetector
from perspective import PerspectiveTransform
from basics import Line


def getVanishingPoint(lines, distance=50, iterations=100):
    """
    Calculate the vanishing point of the road markers by using RANSAC algo.
    :param lines: the lines defined as a [x1, y1, x2, y2] (4xN array, where N is the number of lines)
    :param distance: the distance (in pixels) to determine if a measurement is consistent
    :param iterations: the number of RANSAC iterations to use
    :return: Coordinates of the road vanishing point
    """

    # Number of lines
    n = len(lines)

    # Maximum number of consistant lines
    max_num_consistent_lines = 0

    # Best fit point
    best_fit = None

    isNumpy = isinstance(lines, np.ndarray)

    # Loop through all of the iterations to find the most consistent value
    for i in xrange(iterations):

        # Randomly choosing the lines
        random_indices = np.random.choice(n, 2, replace=False)
        i1 = random_indices[0]
        i2 = random_indices[1]

        l1 = lines[i1][0] if isNumpy else lines[i1]
        l2 = lines[i2][0] if isNumpy else lines[i2]
        # print l1
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        # Find the intersection point
        try:
            x_intersect = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            y_intersect = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        except ZeroDivisionError:
            continue

        if y_intersect < 80 or y_intersect > 300:
            continue

        this_num_consistent_lines = 0

        # Find the distance between the intersection and all of the other lines
        for i2 in range(0, n):
            l2 = lines[i2][0] if isNumpy else lines[i2]
            tx1, ty1, tx2, ty2 = l2
            this_distance = (np.abs((ty2 - ty1) * x_intersect - (tx2 - tx1) * y_intersect + tx2 * ty1 - ty2 * tx1)
                             / np.sqrt((ty2 - ty1) ** 2 + (tx2 - tx1) ** 2))

            if this_distance < distance:
                this_num_consistent_lines += 1

        # If it's greater, make this the new x, y intersect
        if this_num_consistent_lines > max_num_consistent_lines:
            best_fit = int(x_intersect), int(y_intersect)
            max_num_consistent_lines = this_num_consistent_lines

    return best_fit


def parseArguments():
    """
    Parse the command line.
    @return a list of arguments
    """
    parser = argparse.ArgumentParser(
        description="Defines a region of interest.", )

    parser.add_argument("-i", "--image", dest="image",
                        help="input source path")

    return parser.parse_args()


def nothing(position):
    # print position
    pass


def setYBounds(image):
    """
    Define a strip of interest by retrieving the Y bounds.
    :param image: image to process
    :return: a tuple with Y bounds
    """
    height, width = image.shape[:2]
    yMin, yMax = height // 2, height * 95 // 100

    cv2.namedWindow('ROI')
    cv2.createTrackbar('Top', 'ROI', yMin, height, nothing)
    cv2.createTrackbar('Bottom', 'ROI', yMax, height, nothing)

    while (1):
        output = image.copy()

        yMin = cv2.getTrackbarPos('Top', 'ROI')
        yMax = cv2.getTrackbarPos('Bottom', 'ROI')
        cv2.line(output, (0, yMin), (width - 1, yMin), (0, 255, 255), 1)
        cv2.line(output, (0, yMax), (width - 1, yMax), (0, 255, 255), 1)

        cv2.imshow('ROI', output)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return yMin, yMax


def getTrapezoid(image, yBounds):
    # Define a region of interest (ROI)
    height, width = image.shape[:2]

    # Convert and filter the ROI
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayed[0:yBounds[0]] = 0
    grayed[yBounds[1]:] = 0
    cv2.imshow("grayed", grayed)
    equalized = cv2.equalizeHist(grayed)
    _, threshold = cv2.threshold(equalized, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imshow("threshold", threshold)

    # Detect all lines within this ROI
    lsd = LineSegmentDetector()
    npLines = lsd.start(threshold)

    # Process those detected lines to
    MIN_ANGLE = 10. / 180 * np.pi
    MAX_ANGLE = 90. / 180 * np.pi
    xSeparator = width // 2

    leftLines, rightLines = [], []
    for npPoints in npLines:
        points = np.int_(npPoints).tolist()[0]
        line = Line(points)
        slope, _ = line.getSlopeIntercept()
        if slope is None:
            continue

        if (slope > MIN_ANGLE) and (slope < MAX_ANGLE):
            if line.getXMax() > xSeparator:
                rightLines.append(line)
        elif (slope < -MIN_ANGLE) and (slope > -MAX_ANGLE):
            if line.getXMin() < xSeparator:
                leftLines.append(line)

    lines = [line.getPoints(True) for line in leftLines + rightLines]

    trapezoid, vp = None, None

    if leftLines or rightLines:
        # Retrieve the left and right edge of the road
        getNearest = lambda line: line.getYMax()
        leftLines = sorted(leftLines, key=getNearest, reverse=True)
        rightLines = sorted(rightLines, key=getNearest, reverse=True)

        tolerance = 0.10  # percentage
        maxDistance = width * tolerance
        vp = getVanishingPoint(lines, height // 20)
        isNearVP = lambda l: l if np.fabs(l.getX(vp[1]) - vp[0]) < maxDistance else None
        leftInliers = filter(None, [isNearVP(l) for l in leftLines])
        rightInliers = filter(None, [isNearVP(l) for l in rightLines])

        if not leftInliers:
            raise Exception("no left inliers")
        if not rightInliers:
            raise Exception("no right inliers")

        leftEdge = leftInliers[0]
        rightEdge = rightInliers[0]

        # Define the trapezoid
        trapezoid = (int(leftEdge.getX(yBounds[1])), yBounds[1],
                     int(leftEdge.getX(yBounds[0])), yBounds[0],
                     int(rightEdge.getX(yBounds[0])), yBounds[0],
                     int(rightEdge.getX(yBounds[1])), yBounds[1])

    # output = np.zeros_like(source)
    # output = roi.copy()

    # Draw filtered lines
    # lsd.draw(output, lines)

    return threshold, lines, trapezoid, vp


def setTrapezoid(image, threshold, lines, trapezoid, vp):
    height, width = image.shape[:2]

    print 'trapezoid=', trapezoid

    # Display all the detected lines
    for line in lines:
        cv2.line(image, tuple(line[:2]), tuple(line[2:4]), (255, 0, 0), 1)

    # Display the vanishing point
    cv2.circle(image, tuple(vp), 5, (0, 0, 255), -1)

    # Define the track bar
    vertices = [list(pt) for pt in zip(trapezoid[::2], trapezoid[1::2])]
    cv2.namedWindow('Trapezoid',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trapezoid', 980, 640)
    cv2.createTrackbar('X1', 'Trapezoid', vertices[0][0], width * 2, nothing)
    cv2.createTrackbar('X2', 'Trapezoid', vertices[1][0], width, nothing)
    cv2.createTrackbar('X3', 'Trapezoid', vertices[2][0], width, nothing)
    cv2.createTrackbar('X4', 'Trapezoid', vertices[3][0], width * 2, nothing)

    while True:
        # output = image.copy()
        vertices[0][0] = cv2.getTrackbarPos('X1', 'Trapezoid')
        vertices[1][0] = cv2.getTrackbarPos('X2', 'Trapezoid')
        vertices[2][0] = cv2.getTrackbarPos('X3', 'Trapezoid')
        vertices[3][0] = cv2.getTrackbarPos('X4', 'Trapezoid')

        # cv2.line(output, (bottomL, trapezoid[1]), (topL, trapezoid[3]), (0, 255, 255), 2)
        # cv2.line(output, (topR, trapezoid[5]), (bottomR, trapezoid[7]), (0, 255, 255), 2)
        pts = np.array(vertices)
        layer = np.zeros_like(image)
        cv2.fillPoly(layer, [pts, ], (0, 255, 255))
        output = cv2.addWeighted(image.copy(), 1., layer, 0.2, 0.)

        perspective = PerspectiveTransform(threshold, vertices)
        warped = cv2.cvtColor(perspective.warp(threshold), cv2.COLOR_GRAY2BGR)

        cv2.imshow('Trapezoid', np.vstack((output, warped)))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return vertices


def detectROI():
    """
    main function
    :return:
    """
    args = parseArguments()
    path = args.image
    image = cv2.imread(path)
    print 'Press Esc key to escape/continue...'
    print 'Image: %s %s' % (path, str(image.shape))

    g.init()
    # Define the trapezoid
    yBounds = setYBounds(image)
    if not yBounds:
        raise Exception('Fail to define Y bounds')

    threshold, lines, trapezoid, vp = getTrapezoid(image, yBounds)
    trapezoid = setTrapezoid(image, threshold, lines, trapezoid, vp)

    # Save the trapezoid vertices into a file (serialization)
    pos = path.rfind('.')
    if pos != -1:
        trapezoid_file = path[:pos] + '.tpz'
        with open(trapezoid_file, 'wb') as h:
            pickle.dump(trapezoid, h, protocol=2)

    cv2.destroyAllWindows()


# ----------------------  MAIN  ----------------------------
if __name__ == '__main__':
    detectROI()

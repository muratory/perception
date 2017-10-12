#!/usr/bin/env python
# -*- coding: utf-8 -

"""
-------------------------------------------------------------------------------

    Module: basics

    Description:
        enter_a_module_description.
-------------------------------------------------------------------------------

    Copyright (c) 2017, Intel Corporation All Rights Reserved. 


History:
Author          Date         CR           Description of Changes
-------------------------------------------------------------------------------
gcosquer     Jan 20, 2017     bug     Creation
-------------------------------------------------------------------------------
"""

# ----------------------  IMPORTS  ----------------------------
from __builtin__ import str
from _ctypes import ArgumentError
from cmath import atan, cos, sin
from math import sqrt, fabs

import cv2
import numpy as np
from numpy import real


# ----------------------  CLASSES  ----------------------------
# class Point(object):
#     """
#     Class that handles
#     """
#     def __init__(self, x, y):
#         self.__x = x;
#         self.__y = y;
#         
#     def __str__(self, *args, **kwargs):
#         return "Point(%d, %d)" % (self.__x, self.__y)
#     
#     @property
#     def x(self):
#         return self.__x
#     
#     @property
#     def y(self):
#         return self.__y
#     
#     def toArray(self):
#         return (self.__x, self.__y)
# .........................................................
class Line(object):
    """
    Class that handles a line with cartesian coordinates.
    """

    def __init__(self, points):
        """
        Initialize.
        @param points: 4-element tuple defining the line
        """
        if (not hasattr(points, '__iter__')) or (len(points) != 4):
            raise ArgumentError('4-element tuple or list is expected!')

        self.__points = points
        self.__slopeIntercept = None
        self.__angle = None

    def __str__(self, *args, **kwargs):
        return 'Line: ' + str(self.__points)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    @property
    def pt1(self):
        return tuple(self.__points[0:2])

    @property
    def pt2(self):
        return tuple(self.__points[2:4])

    def __computeSlopeIntercept(self):
        """
        Compute the slope and the intercept.
        """
        try:
            a = 1. * (self.__points[3] - self.__points[1]) / (self.__points[2] - self.__points[0])
            self.__slopeIntercept = [a, self.pt1[1] - (a * self.pt1[0])]
        except ZeroDivisionError:
            self.__slopeIntercept = [None, self.__points[0]]

    def draw(self, image, color=(255, 0, 0), thickness=1):
        """
        Draw the current line.
        @param image the input image
        @param color the color of the lineto draw
        @param thickness the thickness of the line to draw in pixels
        """
        cv2.line(image, self.pt1, self.pt2, color, thickness)

    def getLength(self):
        """
        Return the length of the current line
        """
        return sqrt(((self.__points[2] - self.__points[0]) ** 2) + ((self.__points[3] - self.__points[1]) ** 2))



    def getSlopeIntercept(self):
        """
        @return: a tuple with the slope and the intercept
        """
        if self.__slopeIntercept is None:
            self.__computeSlopeIntercept()

        return tuple(self.__slopeIntercept)

    def getAngle(self):
        """
        @return: the current angle in radian.
        """
        if self.__angle is not None:
            return self.__angle

        slope = self.getSlopeIntercept()[0]
        if slope is None:
            self.__angle = np.pi / 2.
        else:
            self.__angle = real(atan(slope)).take(0)
        return self.__angle

    def getIntersectionPoint(self, otherLine):
        """
        Compute the intersection point of 2 lines.
        :param otherLine: line to compute
        :return: the intersection point or None
        """
        slope1, intercept1 = self.getSlopeIntercept()
        slope2, intercept2 = otherLine.getSlopeIntercept()
        if slope1 is None:
            x = intercept1
            y = otherLine.getY(x)
        elif slope2 is None:
            x = intercept2
            y = self.getY(x)
        elif slope1 == slope2:
            return None
        else:
            x = (intercept2 - intercept1) / (slope1 - slope2)
            y = self.getY(x)

        return int(x), int(y)

    def getPoints(self, flattened=False):
        if flattened:
            return tuple(self.__points)
        return tuple(self.__points[0:2]), tuple(self.__points[2:4])

    def getX(self, y):
        """
        Retrieve the X-coordinate.
        @param y the input Y-coordinate
        @return the X-coordinate
        """
        slope, intercept = self.getSlopeIntercept()

        if slope == 0:
            raise Exception('unable to determine the X-coordinate: parallel to X-axis (slope is null).')

        if slope is None:
            return intercept

        return (y - intercept) / slope

    def getY(self, x):
        """
        Retrieve the X-coordinate.
        @param x the input X-coordinate
        @return the Y-coordinate
        """
        slope, intercept = self.getSlopeIntercept()

        if slope is None:
            raise Exception('unable to determine the Y-coordinate: parallel to Y-axis.')

        return (slope * x) + intercept

    def getXMax(self):
        return max(self.__points[0], self.__points[2])

    def getYMax(self):
        return max(self.__points[1], self.__points[3])

    def getXMin(self):
        return min(self.__points[0], self.__points[2])

    def getYMin(self):
        return min(self.__points[1], self.__points[3])

    def isSimilar(self, otherLine, point, maxShift):
        """

        :param otherLine:
        :param point:
        :param maxShift:
        :return:
        """
        # Check the validity of the line to compare
        if otherLine is None:
            return None

        # Check the angle
        MAX_ANGLE = np.pi * 0.025 # ~5°
        if fabs(self.getAngle() - otherLine.getAngle()) > MAX_ANGLE:
            return False

        # Check the proximity at a specific point
        if maxShift[0] is not None:
            if fabs(self.getX(point[1]) - otherLine.getX(point[1])) > maxShift[0]:
                return False

        if maxShift[1] is not None:
            if fabs(self.getY(point[0]) - otherLine.getX(point[0])) > maxShift[1]:
                return False

        return True


# .........................................................
class LinePolar(object):
    """
    Class that handles a line with polar coordinates.
    """
    def __init__(self, rho, theta):
        """
        Initialize.

        :param rho: the distance from origin to the intersection of the perpendicular
        :param theta: the angle in radian
        """
        self.__rho = rho
        self.__theta = theta
        self.__cosinus = cos(theta)
        self.__sinus = sin(theta)

    def __str__(self, *args, **kwargs):
        return 'Line: ' + str(self.__rho) + ' <-> ' + str(self.__theta)

    def draw(self, image, color=(255, 0, 0), thickness=1):
        """
        Draw the current line.
        @param image the input image
        @param color the color of the line to draw
        @param thickness the thickness of the line to draw in pixels
        """
        pt1 = self.getPoint(0)
        pt2 = self.getPoint(image.shape[0])
        cv2.line(image, pt1, pt2, color, thickness)

    def getPoint(self, y):
        return int((self.__rho - (y * self.__sinus)) / self.__cosinus), y


# .........................................................
class Polygon(object):
    """
    Class that handles a polygon
    """

    def __init__(self, vertices=None):
        """
        Initialize.
        
        @param vertices: array containing list of vertices in (x,y) coordinates
        """
        self.__vertices = []

        if vertices is None:
            return

        if hasattr(vertices, '__iter__'):
            self.__vertices.extend(vertices)
        else:
            raise ArgumentError('an iterable arguments was expected (list or tuple)!')

    def __str__(self, *args, **kwargs):
        return 'Polygone: ' + str(self.__vertices)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

    def addVertex(self, vertex):
        """
        Append a vertex to the current polygon.
        @param vertex: list or tuple containing vertex coordinates
        """
        self.__vertices.append(vertex)

    def draw(self, image, color=(0, 0, 255), alpha=None):
        """
        Draw the polygon.
        :param image: the output image
        :param color: the color of lines (red by default)
        :param alpha: the transparency parameter (default: None)
        """
        pts = np.array(self.__vertices)

        if alpha is None:
            cv2.fillPoly(image, [pts, ], color)
            output = image
        else:
            layer = np.zeros_like(image)
            cv2.fillPoly(layer, [pts, ], color)
            output = cv2.addWeighted(image.copy(), 1, layer, alpha, 0.)

        return output

    def getSideCount(self):
        """
        @return: the number of vertices of the current polygon.
        """
        return len(self.__vertices)


# ----------------------  FUNCTION  ------------------------
def test():
    print cv2.__version__
    m = np.zeros((386, 640, 3))
    l1 = Line((50, 50, 3, 100))
    print 'l1=', l1
    print 'length=', l1.getLength()
    print 'slope, intercept=', l1.getSlopeIntercept()
    print 'angle=', l1.getAngle() / np.pi * 180.
    print "xmax=", l1.getXMax()

    l2 = Line((2, 2, 100, 10))
    print 'l2=', l2
    print 'length=', l2.getLength()
    print 'slope, intercept=', l2.getSlopeIntercept()
    print 'angle=', l2.getAngle() / np.pi * 180.
    print "xmax=", l2.getXMax()

    print "l1 U l2=", l1.getIntersectionPoint(l2), '=', l2.getIntersectionPoint(l1)

    #x = 5.
    #y = l1.getY(x)
    #     print  x, '->', y, "->", l1.getX(y)
    l1.draw(m)
    l2.draw(m)

    p = Polygon(((10, 10), (10, 50), (50, 50)))
    p.addVertex((50, 10))
    p.addVertex((40, 20))
    m = p.draw(m)

    p = Polygon(((20, 5), (25, 50), (40, 40)))
    m = p.draw(m, color=(0, 255, 0), alpha=0.1)

    n = (None, Line((598, 324, 626, 332)))
    l = (Line((42, 242, 72, 238)), Line((590, 296, 636, 322)))
    height, width = 386, 640
    maxShift = (width * 0.05, None)
    print l[1].isSimilar(n[1],(None, height), maxShift )
    print l[1].getX(height), 'vs', n[1].getX(height)
    l[1].draw(m)
    n[1].draw(m)

    cv2.imshow('test', m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------  MAIN  ----------------------------
if __name__ == '__main__':
    test()

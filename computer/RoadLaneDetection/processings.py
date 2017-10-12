#!/usr/bin/env python
# -*- coding: utf-8 -

"""
-------------------------------------------------------------------------------

    Module: processings

    Description:
        This modules deals with image processings
-------------------------------------------------------------------------------

    Copyright (c) 2017, All Rights Reserved.


History:
Author          Date         Issue         Description of Changes
-------------------------------------------------------------------------------
gcosquer     Nov 10, 2016     ---       Creation
-------------------------------------------------------------------------------
"""

# ----------------------  IMPORTS  ----------------------------
import cv2
import numpy as np


# ----------------------  INTERFACES  ----------------------------
class ImageProcessing(object):
    def initialize(self):
        raise NotImplementedError('This method must be define in any derivatives')

    def start(self, source):
        raise NotImplementedError('This method must be define in any derivatives')

    def terminate(self):
        raise NotImplementedError('This method must be define in any derivatives')


# ----------------------  CLASSES  ----------------------------

# ............................................................
class CannyEdgeDetector(ImageProcessing):
    """
    Finds edges in an image using the [Canny86] algorithm.
    """
    THRESHOLD1 = 120
    THRESHOLD2 = 180

    def __init__(self, threshold1=THRESHOLD1, threshold2=THRESHOLD2):
        """
        Initialize.
        """
        self.__threshold1 = threshold1
        self.__threshold2 = threshold2
        self.__aperture = 3

    def initialize(self):
        pass

    def start(self, source):
        return cv2.Canny(source, self.__threshold1, self.__threshold2, self.__aperture)

    def terminate(self):
        pass

    def setThresholds(self, threshold1, threshold2):
        self.__threshold1 = threshold1
        self.__threshold2 = threshold2


# ............................................................        
class HoughLineDetector(ImageProcessing):
    """
    Finds edges in an image using the [Canny86] algorithm.
    """
    VOTE = 20
    MIN_LINE_LENGTH = 10.
    MAX_LINE_GAP = 10.

    def __init__(self):
        """
        Initialize
        """
        pass

    def initialize(self):
        pass

    #     def stop(self):
    #         raise NotImplementedError

    def start(self, source):
        """
        @return: lines
        """
        vote = self.VOTE
        minLineLength = self.MIN_LINE_LENGTH
        maxLineGap = self.MAX_LINE_GAP
        lines = cv2.HoughLinesP(source, 1, np.pi / 180., vote, minLineLength, maxLineGap)
        return lines

    def draw(self, image, lines, color=(255, 0, 0), thickness=2):
        """
        Draw lines into the original image.
        """
        for i, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                # cv2.putText(image, str(i), (int(x1) - 10, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def terminate(self):
        pass


# ............................................................        
class LineSegmentDetector(ImageProcessing):
    """
    Finds edges in an image using the [Canny86] algorithm.
    """
    def __init__(self):
        """
        Initialize
        """
        self.__lsd = cv2.createLineSegmentDetector()

    def initialize(self):
        pass

    def start(self, source):
        segments = self.__lsd.detect(source)
        if segments is not None:
            return segments[0]
        return None

    def terminate(self):
        pass

    def draw(self, image, segments, color=(255, 0, 0), thickness=1):
        """
        Draw all the segments.
        :param image: source image
        :param segments: list of segments
        :param color: color of the line (red by default)
        :param thickness:
        :return:
        """
        #         self.__lsd.drawSegments(image, segments)

        if isinstance(segments, np.ndarray):
            for segment in segments:
                for x1, y1, x2, y2 in segment:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        else:
            for segment in segments:
                cv2.line(image, tuple(segment[:2]), tuple(segment[2:4]), color, thickness)


# ----------------------  MAIN  -------------------------------
if __name__ == '__main__':
    pass

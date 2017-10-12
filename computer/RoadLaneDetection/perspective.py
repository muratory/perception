#!/usr/bin/env python
# -*- coding: utf-8 -

"""
-------------------------------------------------------------------------------

    Module: perspective

    Description:
        enter_a_module_description.
-------------------------------------------------------------------------------

    Copyright (c) 2017 , All Rights Reserved.


History:
Author          Date         Issue      Description of Changes
-------------------------------------------------------------------------------
gcosquer      3/31/2017      ---        Creation
-------------------------------------------------------------------------------
"""

# ----------------------  IMPORTS  -------------------------
import cv2
import pickle
from numpy import float32


# ----------------------  CLASSES  -------------------------
class PerspectiveTransform(object):
    """
    Class that manages perspective transformation by using opencv
    """

    def __init__(self, image, src, ratio=0.25):
        """
        Initialize.
        :param src: coordinates of quadrangle vertices in the source image
        :param ratio: coordinates of the corresponding quadrangle vertices in the destination image.
        """
        height, width = image.shape[:2]

        # Calculate the perspective transform and its inverse transform
        # from four pairs of the corresponding points
        src = float32(src)
        dst = float32(
            [[ratio * width, height],
             [ratio * width, 0],
             [(1. - ratio) * width, 0],
             [(1. - ratio) * width, height]])

        self.__transform = cv2.getPerspectiveTransform(src, dst)
        self.__inv_transform = cv2.getPerspectiveTransform(dst, src)

    def save(self, path):
        """
        Save both transform matrices into a binary file by using pickle (serialization)
        :param path: file path to save
        """
        with open(path, 'wb') as h:
            pickle.dump((self.__transform, self.__inv_transform), h, protocol=2)

    def load(self, path):
        """
        Load both transform matrices from a binary file by using pickle (deserialization)
        :param path: file path to save
        :return:
        """
        with open(path, 'rb') as h:
            self.__transform, self.__inv_transform = pickle.load(h)

    def warp(self, image):
        """
        Apply a perspective transformation to an image.
        :return: the transformed image
        """
        height, width = image.shape[:2]
        return cv2.warpPerspective(image, self.__transform, (width, height), flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        """
        Apply an inverse perspective transformation to an image.
        :return: the transformed image
        """
        height, width = image.shape[:2]
        return cv2.warpPerspective(image, self.__inv_transform, (width, height), flags=cv2.INTER_LINEAR)


# ----------------------  MAIN  ----------------------------
if __name__ == '__main__':
    pass

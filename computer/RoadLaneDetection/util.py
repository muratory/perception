import cv2
from scipy.misc import imshow
from laneconfiguration import *
import global_vars as g

def display_debug_image(img, frame_name, position=(0,0), param_dict=None, range_dict=None):
    """
    Display the image in parameter in a window according to the debug level.
    If trackbars are enabled, then param_dict is used to describe the trackbars to be created.

    :param img: image to display
    :param frame_name: name of the window
    :param position: position of the window, tuple (x, y)
    :param param_dict: dictionary of parameters param_name => value to use to create trackbars
    :param range_dict: dictionary of ranges for parameters param_name => value range
    :return:
    """
    if DEBUG_IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if img != None:
            imshow(img)
    if DEBUG_CV2IMSHOW_LEVEL >= DEBUG_LEVEL2:
        if g.cold_boot:
            # Create the window
            cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(frame_name, g.s_win_width, g.s_win_height)
            cv2.moveWindow(frame_name, position[0] * g.s_win_width, position[1] * g.s_win_height)

            # Create the trackbars
            if g.trackbar_enabled and param_dict != None:
                for param_name, param_value in param_dict.items():
                    if range_dict != None:
                        param_min, param_max = range_dict[param_name]
                    else:
                        param_min, param_max = 0, 255
                    cv2.createTrackbar(param_name, frame_name, param_min, param_max, lambda x: x)
                    cv2.setTrackbarPos(param_name, frame_name, param_value)

        # Update parameters with the new value from trackbars
        if g.trackbar_enabled and param_dict != None:
            for param_name in param_dict.keys():
                param_dict[param_name] = cv2.getTrackbarPos(param_name, frame_name)

        # Display image
        if img is not None:
            cv2.imshow(frame_name, img)
        cv2.waitKey(1)

def display_debug_trackers(frame_name, param_dict, range_dict=None, position=(0,0)):
    """
    Display a window containing only trackbars and no image.

    :param frame_name: window name
    :param param_dict:
    :param position: dictionary of parameters param_name => value to use to create trackbars
    :param range_dict: dictionary of ranges for parameters param_name => value range
    :return:
    """
    display_debug_image(None, frame_name=frame_name, param_dict=param_dict, range_dict=range_dict, position=position)

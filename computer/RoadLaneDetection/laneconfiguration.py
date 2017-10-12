""" DISTORSION CORRECTION """
#RLD_ZERO_DISTORSION=False
RLD_ZERO_DISTORSION = True

""" SCREEN TEXT ANNOTATION """
VI_HEIGHT = 720
VI_FONT_SIZE = 0.65
VI_TEXT_OFFSET = 20
TI_KW = 1
TI_PLT = 0.001

""" ROAD DEFINITION PARAMS """
# 3,5 meter for each lane (French reference: http://www.securite-routiere-az.fr/v/voie-de-circulation/ )
LANE_WIDTH = 3500.
# Histogram windows size set to +/- 500mm (50cm)
HIST_WIN_WIDTH = 500.
# Number of Windows for Histogramme computation
N_WINDOWS = 9

""" DECTECTION SLOW/FAST MODE PARAMETERS """
SLOW_MODE = 'SLOW_MODE'
FAST_MODE = 'FAST_MODE'

DEGRAD_VIZ_THRESHOLD = 12 # In Fast Mode Degrad Viz Th Frames are allowed

""" ROI TRAPEZE AREA REGION OF INTERREST FOR LANE DETECTION """
# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
TRAP_BOTTOM_WIDTH = 0.76                                                 # (0.7 -> 0.6) width of bottom edge of trapezoid, expressed as percentage of image width
TRAP_TOP_WIDTH = 0.10                                                    # (0.07 -> 0.10) ditto for top edge of trapezoid
TRAP_HEIGHT = 0.36                                                       # (0.4 -> 0.37 -> 0.3) height of the trapezoid expressed as percentage of image height
TRAP_WARPED_RATIO=TRAP_BOTTOM_WIDTH-TRAP_TOP_WIDTH                       # (0,53 -> 0,60) used for perspective transform computation

# Challenge
TRAP_BOTTOM_WIDTH_CFG1 = 0.70                                            # (0.7 -> 0.6) width of bottom edge of trapezoid, expressed as percentage of image width
TRAP_TOP_WIDTH_CFG1 = 0.15    #0.2                                       # (0.07 -> 0.10) ditto for top edge of trapezoid
TRAP_HEIGHT_CFG1 = 0.30       #0.28                                      # (0.4 -> 0.37 -> 0.3) height of the trapezoid expressed as percentage of image height
TRAP_WARPED_RATIO_CFG1=TRAP_BOTTOM_WIDTH_CFG1-TRAP_TOP_WIDTH_CFG1        # (0,53 -> 0,60) used for perspective transform computation

# Toulouse
TRAP_BOTTOM_WIDTH_CFG2 = 0.95                                            # (0.7 -> 0.6) width of bottom edge of trapezoid, expressed as percentage of image width
TRAP_TOP_WIDTH_CFG2 = 0.20                                               # (0.07 -> 0.10) ditto for top edge of trapezoid
TRAP_HEIGHT_CFG2 = 0.40                                                  # (0.4 -> 0.37 -> 0.3) height of the trapezoid expressed as percentage of image height
TRAP_WARPED_RATIO_CFG2=TRAP_BOTTOM_WIDTH_CFG2-TRAP_TOP_WIDTH_CFG2        # (0,53 -> 0,60) used for perspective transform computation

# Toulouse
TRAP_BOTTOM_WIDTH_CFG2_1 = 0.93                                          # (0.7 -> 0.6) width of bottom edge of trapezoid, expressed as percentage of image width
TRAP_TOP_WIDTH_CFG2_1 = 0.24                                             # (0.07 -> 0.10) ditto for top edge of trapezoid
TRAP_HEIGHT_CFG2_1 = 0.32                                                # (0.4 -> 0.37 -> 0.3) height of the trapezoid expressed as percentage of image height
TRAP_WARPED_RATIO_CFG2_1=TRAP_BOTTOM_WIDTH_CFG2_1-TRAP_TOP_WIDTH_CFG2_1  # (0,53 -> 0,60) used for perspective transform computation


# Nice
TRAP_BOTTOM_WIDTH_CFG3 = 0.80                                            # (0.7 -> 0.6) width of bottom edge of trapezoid, expressed as percentage of image width
TRAP_TOP_WIDTH_CFG3 = 0.10                                               # (0.07 -> 0.10) ditto for top edge of trapezoid
TRAP_HEIGHT_CFG3 = 0.35                                                  # (0.4 -> 0.37 -> 0.3) height of the trapezoid expressed as percentage of image height
TRAP_WARPED_RATIO_CFG3=TRAP_BOTTOM_WIDTH_CFG3-TRAP_TOP_WIDTH_CFG3        # (0,53 -> 0,60) used for perspective transform computation


""" FILTER COMBINED MODE """
FILTER_DEFAULT = 'DEFAULT'
FILTER_HSV_HSL = 'HLS_OR_HSV'
FILTER_COLOR = 'COLOR'
FILTER_EQUALIZ = 'EQUALIZ'

FILTER_GRAD_MAG = 'GRAD_MAG'
FILTER_GRAD_MAG_DIR = 'GRAD_MAG_DIR'
FILTER_GRAD_MAG_HSV = 'GRAD_MAG_HSV'

FILTER_MAG_DIR_HSV = 'MAG_DIR_HSV'
FILTER_MAG_DIR_HLS_HSV = 'MAG_DIR_HLS_HSV'

FILTER_SOBEL_MAG_DIR_HLS_HSV = 'SOBEL_MAG_DIR_HLS_HSV'

""" DEBUG CONFIGURATION FOR LOGGING """
DEBUG_LEVEL0 = 0
DEBUG_LEVEL1 = 1
DEBUG_LEVEL2 = 2
DEBUG_LEVEL3 = 3

DEBUG_LEVEL_LIST = [DEBUG_LEVEL0, DEBUG_LEVEL1, DEBUG_LEVEL2, DEBUG_LEVEL3]

DEBUG_LEVEL = DEBUG_LEVEL_LIST[1]
DEBUG_IMSHOW_LEVEL = DEBUG_LEVEL_LIST[0]
DEBUG_CV2IMSHOW_LEVEL = DEBUG_LEVEL_LIST[0]
DEBUG_COMBINED_THRESHOLD = DEBUG_LEVEL_LIST[0]
DEBUG_LINE_FIT = DEBUG_LEVEL_LIST[0]
DEBUG_PLT_LINE_FIT = DEBUG_LEVEL_LIST[0]
DEBUG_LINE_FIT_VIDEO = DEBUG_LEVEL_LIST[0]
DEBUG_PLT_LINE_FIT_VIDEO = DEBUG_LEVEL_LIST[0]


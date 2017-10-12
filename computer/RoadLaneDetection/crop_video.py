from __future__ import division
from __future__ import print_function


import cv2
import traceback

from moviepy.editor import VideoFileClip

global x_coord, y_coord, x_width, y_height


# Helper functions
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


def get_fl_name():
    stack = traceback.extract_stack()
    filename, codeline, funcName, text = stack[-2]
    return (funcName,codeline)


def crop_video(img_in):
    """ Given an image Numpy array, return the cropped image as a Numpy array """
    global x_coord, y_coord, x_width, y_height
    (h,w) = (img_in.shape[0], img_in.shape[1])
    crop_img = img_in[y_coord:h-y_coord-y_height, x_coord:w-x_coord-x_width] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    return crop_img


def crop_video_file(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(crop_video)
    annotated_video.preview()
    annotated_video.write_videofile(output_file, codec='libx264', audio=False)

    """
    codec
    Codec to use for image encoding. Can be any codec supported
    by ffmpeg. If the filename is has extension '.mp4', '.ogv', '.webm',
    the codec will be set accordingly, but you can still set it if you
    don't like the default. For other extensions, the output filename
    must be set accordingly.
    Some examples of codecs are:
    ``'libx264'`` (default codec for file extension ``.mp4``)
    makes well-compressed videos (quality tunable using 'bitrate').
    ``'mpeg4'`` (other codec for extension ``.mp4``) can be an alternative
    to ``'libx264'``, and produces higher quality videos by default.
    ``'rawvideo'`` (use file extension ``.avi``) will produce
    a video of perfect quality, of possibly very huge size.
    ``png`` (use file extension ``.avi``) will produce a video
    of perfect quality, of smaller size than with ``rawvideo``
    ``'libvorbis'`` (use file extension ``.ogv``) is a nice video
    format, which is completely free/ open source. However not
    everyone has the codecs installed by default on their machine.
    ``'libvpx'`` (use file extension ``.webm``) is tiny a video
    format well indicated for web videos (with HTML5). Open source.
    audio
    Either ``True``, ``False``, or a file name.
    If ``True`` and the clip has an audio clip attached, this
    audio clip will be incorporated as a soundtrack in the movie.
    If ``audio`` is the name of an audio file, this audio file
    will be incorporated as a soundtrack in the movie.
    """

# End helper functions

def main():
    """Main function"""
    # construct the argument parse and parse the arguments

    from optparse import OptionParser
    global x_coord, y_coord, x_width, y_height


    # Configure command line options
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                      help="Input video/image file")
    parser.add_option("-o", "--output_file", dest="output_file",
                      help="Output (destination) video/image file")

    parser.add_option("-c", "--coord", dest="coord", nargs=4, type=int,
                      help="x y w h --> Crop from x, y, w, h")

    # Get and parse command line options
    options, args = parser.parse_args()
    print ('options:', options)
    print ('args:', args)

    if 'input_file' not in options.__dict__:
        parser.error('Input Filename not given')
    if options.input_file is None:
        parser.error('Input Filename not given')
    if 'output_file' not in options.__dict__:
        parser.error('Output Filename not given')
    if options.output_file is None:
        parser.error('Output Filename not given')
    if 'coord' not in options.__dict__:
        parser.error('4 Coord not given')
    if options.coord is None:
        parser.error('Coord are not defined ')

    input_file = options.input_file
    output_file = options.output_file
    coord = options.coord
    #print ('coord:', coord)

    x_coord = coord[0]
    y_coord = coord[1]
    x_width = coord[2]
    y_height = coord[3]
    #print ('x_coord:', x_coord, 'y_coord:', y_coord, 'x_width:', x_width, 'y_height:', y_height)

    crop_video_file(input_file, output_file)


# Main script
if __name__ == '__main__':
    main()

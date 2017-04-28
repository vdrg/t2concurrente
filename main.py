import sys
import getopt
import imageio
import equalizer
import serial_equalizer
from moviepy.editor import *


def usage():
    print("Usage: python program src dest")

def main(argv = None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 3:
        usage()
        sys.exit(2)

    

    # Load video
    clip = VideoFileClip(argv[1]).subclip(0, 17)
    clip.set_fps(5)


    # Number of bins
    BINS = 1024
    edited = clip.fl_image( equalizer.process(BINS) )

    # Write the result to a file
    edited.write_videofile(argv[2])

if __name__ == "__main__":
    main()


#!/usr/bin/python3

import sys
import getopt
from moviepy.editor import *

from serial_equalizer import process as serial_equalize

try:
    from equalizer import process as cuda_equalize
except ImportError:
    print("CUDA not found, using serial equalizer.")


def usage():
    print("Usage: python main.py source destination")
    print("Options:")
    print("  -h, --help       Display this message.")
    print("  -s, --serial     Force serial equalization.")
    print("    , --subclip    Equalize from the beginning of the video until this second. Example: --subclip 10")
    print("    , --verbose    Show aditional information.")

def main(argv = None):
    if argv is None:
        argv = sys.argv

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "hs", ["help", "serial", "subclip=", "verbose"])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    if ("equalizer" not in sys.modules):
        equalize = serial_equalize
    else:
        equalize = cuda_equalize

    verbose = False
    end = 30

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-s", "--serial"):
            equalize = serial_equalize
        elif o == "--subclip":
            end = int(a)
        elif o == "--verbose":
            verbose = True
        else:
            assert False, "Unhandled option"
    
    if len(args) < 2:
        usage()
        sys.exit(2)

    try:
        # Load video
        clip = VideoFileClip(args[0]).subclip(0, end)
        if (clip.fps > 25):
            clip.set_fps(25)


        # Number of bins
        BINS = 256
        edited = clip.fl_image( equalize(BINS, verbose) )

        # Write the result to a file
        edited.write_videofile(args[1])
    except KeyboardInterrupt:
        print("Equalization interrupted.")

if __name__ == "__main__":
    main()


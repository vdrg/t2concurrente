def load_kernels():
    print("Loading and compiling kernels")

    mod = ""
    with open("kernels/histogram.cu", "r") as histogram:
        mod += histogram.read()

    with open("kernels/transform.cu", "r") as transform:
        mod += transform.read()

    with open("kernels/rgb2hsv.cu", "r") as rgb2hsv:
        mod += rgb2hsv.read()

    with open("kernels/hsv2rgb.cu", "r") as hsv2rgb:
        mod += hsv2rgb.read()


    return mod

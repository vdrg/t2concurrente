def load_kernels():
    mod = ""
    with open("histogram.cu", "r") as histogram:
        mod += histogram.read()

    with open("transform.cu", "r") as transform:
        mod += transform.read()

    #  with open()

    return mod

from skimage.color import rgb2hsv, hsv2rgb
import numpy as np

def compute_cdf(hist):
    cdf = np.empty_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
    
def compute_hist(values, bins):
    hist = np.zeros(bins)
    for val in values:
        hist[val] += 1
    return hist

def compute_transform(cdf, size):
    cdf_min = np.amin(cdf)
    transform = np.empty_like(cdf)
    for i in range(len(transform)):
        transform[i] = (cdf[i] - cdf_min) / (size - 1)
    return transform


def process(bins):
    # Currying
    def compute(image):
        print("Moving image to HSV color space.")
        edited = rgb2hsv(image)
        values = edited[:,:,2].flatten() * (bins - 1)
        values = values.round().astype(np.int)

        print("Computing histogram.")
        hist = compute_hist(values, bins)

        print("Computing cdf.")
        cdf = compute_cdf(hist)

        print("Computing transformation.")
        transform = compute_transform(cdf, len(values))

        print("Setting transformed values to the image.")
        for y in range(image.shape[0]):
            currenty = y * image.shape[1]
            for x in range(image.shape[1]):
                edited[y,x,2] = transform[values[x + currenty]]


        #  print(image)
        #  print(np.round(hsv2rgb(edited) * 255))
        print("Moving image back to RGB.")
        return np.round(hsv2rgb(edited) * 255)
    return compute

    # CUDA
    #  img_gpu = cuda.mem_alloc(img.nbytes)
    #  cuda.memcpy_htod(img_gpu, img)

    #  func = mod.get_function("equalize")
    #  func(img_gpu, block=(4,4,1))

    #  img_equalized = np.empty_like(img)
    #  cuda.memcpy_dtoh(img_equalized, img_gpu)




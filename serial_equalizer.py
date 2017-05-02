from skimage.color import rgb2hsv, hsv2rgb
import numpy as np


def compute_cdf(hist):
    cdf = hist.cumsum()
    return cdf / cdf[-1]
    
def compute_hist(values, bins):
    hist = np.zeros(bins)
    for val in values:
        hist[val] += 1
    return hist

def compute_transform(cdf, bins, size):
    cdf_min = np.amin(cdf)
    transform = np.empty_like(cdf)
    for i in range(len(transform)):
        transform[i] = (cdf[i] - cdf_min) / (1 - cdf_min)
    return transform


def process(bins, verbose=False):
    verboseprint = print if verbose else lambda *a, **k: None
    # Currying
    def compute(image):
        verboseprint("Moving image to HSV color space.")
        edited = rgb2hsv(image)

        values = edited[:,:,2].flatten() * (bins - 1)
        values = values.round().astype(np.int)

        verboseprint("Computing histogram.")
        hist = compute_hist(values, bins)

        verboseprint("Computing cdf.")
        cdf = compute_cdf(hist)

        verboseprint("Computing transformation.")
        transform = compute_transform(cdf, bins, len(values))

        verboseprint("Setting transformed values to the image.")
        edited[:,:,2] = transform[values].reshape(edited[:,:,2].shape)

        verboseprint("Moving image back to RGB.")
        return hsv2rgb(edited) * 255
    return compute

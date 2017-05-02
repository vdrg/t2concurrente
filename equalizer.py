from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .kernels.loader import load_kernels

mod = SourceModule(load_kernels())
   
def compute_hist(values, bins):
    hist = np.zeros(bins).astype(np.int32)

    hist_func = mod.get_function("hist")
    block = (128,1,1)
    grid = (int((len(values) + block[0] - 1)/block[0]), 1, 1)

    hist_func(cuda.In(values), np.int32(len(values)), cuda.InOut(hist), np.int32(bins), grid=grid, block=block)

    return hist

def compute_cdf(hist):
    cdf = np.empty_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
 
def compute_transform(cdf, size):
    cdf_min = np.amin(cdf)

    transform = np.empty_like(cdf).astype(np.float32)

    transform_func = mod.get_function("transform")
    block = (128, 1, 1)
    grid = (int((len(transform) + block[0] - 1)/block[0]), 1, 1)
    transform_func(cuda.InOut(transform), np.int32(len(transform)), cuda.In(cdf), np.int32(cdf_min), np.int32(size), grid=grid, block=block)
    return transform

def transform_values(img, values, transform, width, height):
    result = np.empty_like(values).astype(np.float32)

    # Yblocks = width / 16
    # if(width % 16 > 0) Yblocks++
    # Xblocks = height / 16;
    # if(height % 16) Xblocks++;
    # block = (16, 16, 1)
    # grid = (Yblocks, Xblocks, 1)
    block = (128, 1, 1)
    grid = (int((len(values) + block[0] - 1)/block[0]), 1, 1)

    # func = mod.get_function("transform_values")
    # func(cuda.In(values), cuda.InOut(result), cuda.In(transform), np.int32(width), np.int32(height), grid=grid, block=block)

    return transform[values].reshape(img[:,:,2].shape)

def process(bins, verbose=False):
    verboseprint = print if verbose else lambda *a, **k: None

    # Currying
    def compute(image):
        verboseprint("Moving image to HSV color space.")
        edited = rgb2hsv(image)
        values = edited[:,:,2].flatten() * (bins - 1)
        values = values.round().astype(np.int32)

        verboseprint("Computing histogram.")
        hist = compute_hist(values, bins)

        verboseprint("Computing cdf.")
        cdf = compute_cdf(hist)

        verboseprint("Computing transformation.")
        transform = compute_transform(cdf, len(values))

        verboseprint("Setting transformed values to the image.")
        edited[:,:,2] = transform_values(edited, values, transform, edited.shape[1], edited.shape[0])

        verboseprint("Moving image back to RGB.")

        return hsv2rgb(edited) * 255
    return compute

    


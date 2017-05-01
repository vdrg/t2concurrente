from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>
    __global__ void hist(int *L, int size,  int *hist, int n)
    {
        int idx, offset;
	idx = blockIdx.x * blockDim.x + threadIdx.x;
        offset = blockDim.x * gridDim.x;

        for (int i = idx; i < size; i += offset) {
            atomicAdd(&hist[L[i]],1);
        }
    }


    __global__ void transform(float* transform, int length, int *cdf, int cdf_min, int img_size) 
    {
        int idx, offset;
	idx = blockIdx.x * blockDim.x + threadIdx.x;
        offset = blockDim.x * gridDim.x;


        for (int i = idx; i < length; i += offset) 
        {
            transform[i] = (float) (cdf[i] - cdf_min) / (img_size - 1);
        }


    }

    __global__ void transform_values_wrong(int *values, float *result, float *transform, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = x + y * width;
        result[idx] = transform[values[idx]];
    }

    __global__ void transform_values(int *values, float *result, float *transform, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int offset = blockDim.x * gridDim.x;

        for (int i = idx; i < size; i += offset) {
            result[idx] = transform[values[idx]];
        }
    }

""")

def compute_cdf(hist):
    cdf = np.empty_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
    
def compute_hist(values, bins):
    # CUDA
    # values_gpu = cuda.mem_alloc(values.nbytes)
    # cuda.memcpy_htod(values_gpu, values)

    # print(values)
    hist = np.zeros(bins).astype(np.int32)

    hist_func = mod.get_function("hist")
    block = (128,1,1)
    grid = (int((len(values) + block[0] - 1)/block[0]), 1, 1)

    hist_func(cuda.In(values), np.int32(len(values)), cuda.InOut(hist), np.int32(bins), grid=grid, block=block)

    #  cuda.memcpy_dtoh(hist, hist_gpu)

    #  hist = np.zeros(bins)
    #  for val in values:
        #  hist[val] += 1
    return hist

def compute_transform(cdf, size):
    cdf_min = np.amin(cdf)
    #  cdf_gpu = cuda.mem_alloc(cdf.nbytes)
    #  cuda.memcpy_htod(cdf_gpu, cdf)

    transform = np.empty_like(cdf).astype(np.float32)
    #  transform_gpu = cuda.mem_alloc(transform.nbytes)

    transform_func = mod.get_function("transform")
    block = (128, 1, 1)
    grid = (int((len(transform) + block[0] - 1)/block[0]), 1, 1)
    transform_func(cuda.InOut(transform), np.int32(len(transform)), cuda.In(cdf), np.int32(cdf_min), np.int32(size), grid=grid, block=block)
    # print(transform)
    #  cdf_min = np.amin(cdf)
    #  transform = np.empty_like(cdf)
    #  for i in range(len(transform)):
        #  transform[i] = (cdf[i] - cdf_min) / (size - 1)
    return transform

def transform_values(img, values, transform, width, height):
    # result_gpu = cuda.mem_alloc(values.astype(np.float32).nbytes)
    result = np.empty_like(values).astype(np.float32)

    # Yblocks = width / 16
    # if(width % 16 > 0) Yblocks++
    # Xblocks = height / 16;
    # if(height % 16) Xblocks++;
    # block = (16, 16, 1)
    # grid = (Yblocks, Xblocks, 1)
    block = (128, 1, 1)
    grid = (int((len(values) + block[0] - 1)/block[0]), 1, 1)

    #func = mod.get_function("transform_values")
    # func(cuda.In(values), cuda.InOut(result), cuda.In(transform), np.int32(width), np.int32(height), grid=grid, block=block)

    # print(img[:,:,2])
    # cuda.memcpy_dtoh(img[:,:,2], result_gpu)
    
    for y in range(img.shape[0]):
        currenty = y * img.shape[1]
        for x in range(img.shape[1]):
            img[y,x,2] = transform[values[x + currenty]]
            #img[y,x,2] = result[x + currenty]

    # print(img[:,:,2])


def process(bins):
    # Currying
    def compute(image):
        print("Moving image to HSV color space.")
        edited = rgb2hsv(image)
        values = edited[:,:,2].flatten() * (bins - 1)
        values = values.round().astype(np.int32)

        print("Computing histogram.")
        hist = compute_hist(values, bins)

        print("Computing cdf.")
        cdf = compute_cdf(hist)

        print("Computing transformation.")
        transform = compute_transform(cdf, len(values))

        print("Setting transformed values to the image.")
        transform_values(edited, values, transform, edited.shape[1], edited.shape[0])

        print("Moving image back to RGB.")
        return np.round(hsv2rgb(edited) * 255)
    return compute

    


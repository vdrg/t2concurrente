from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import pycuda as cuda

mod = """
    __global__ void hist(float *L, int *hist, int n)
    {
        __shared__ int local[n];
        unsigned int idx = threadIdx.x;
        local[idx] = 0;
        __syncthreads();

        unsigned int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
        atomicAdd(&local[L[pixelIdx]], 1);
        __syncthreads();

        atomicAdd(&(hist[idx], local[idx]));
    }


    __global__ void transformation(float* transform, int length, int *cdf, int cdf_min, int img_size) 
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = idx; i < length; i += blockDim.x * gridDim.x) 
        {
            transform[i] = (cdf[i] - cdf_min) / (img_size - 1);
        }
    }

    __global__ void transform_values(int *values, float *result, float *transform, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdy.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = x + y * width;
        result[idx] = transform[values[idx]];
    }
"""

def compute_cdf(hist):
    cdf = np.empty_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
    
def compute_hist(values, bins):
    # CUDA
    values_gpu = cuda.mem_alloc(values.nbytes)
    cuda.memcpy_htod(values_gpu, values)

    hist = np.zeros(bins)

    hist_func = mod.get_function("hist")
    hist_func(values_gpu, cuda.InOut(hist), bins, block=(4,4,1))

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
    transform_func(cuda.InOut(transform), len(transform), cuda.In(cdf), cdf_min, size)
    #  cdf_min = np.amin(cdf)
    #  transform = np.empty_like(cdf)
    #  for i in range(len(transform)):
        #  transform[i] = (cdf[i] - cdf_min) / (size - 1)
    return transform

def transform_values(img, values, transform, width, height):
    result_gpu = cuda.mem_alloc(values.astype(np.float32).nbytes)

    func = mod.get_function("transform_values")
    func(cuda.In(values), result_gpu, cuda.In(transform), width, height)

    cuda.memcpy_dtoh(img[:,:,2], result_gpu)
    
    #  for y in range(img.shape[0]):
        #  currenty = y * img.shape[1]
        #  for x in range(img.shape[1]):
            #  img[y,x,2] = transform[values[x + currenty]]



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
        transform_values(edited, values, transform, edited.shape[1], edited.shape[0])

        print("Moving image back to RGB.")
        return np.round(hsv2rgb(edited) * 255)
    return compute

    


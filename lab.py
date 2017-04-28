from skimage.color import rgb2lab, lab2rgb
import numpy as np

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


    __global__ void transformation(int* transform, int *cdf, int cdf_min, int bins, int img_size) 
    {
        transform[idx] = (cdf[idx] - cdf_min) * (bins - 1) / (img_size - 1);
    }

    __global__ void equalize(int *L, int *Lout, int *transform)
    {
        Lout[idx] = transform[L[idx]];
    }
"""

def compute_cdf(hist):
    cdf = np.empty_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
    
def compute_hist(L=[]):
    hist = np.zeros(101)
    for l in L:
        hist[l] += 1
    return hist

def compute_transform(cdf, size):
    cdf_min = np.amin(cdf)
    transform = np.empty_like(cdf)
    for i in range(len(transform)):
        transform[i] = (cdf[i] - cdf_min) * 100 / (size - 1)
    return transform


def process(image):
    print("Moving image to Lab color space.")
    imageLab = rgb2lab(image)
    L = imageLab[:,:,0].flatten().round().astype(np.int32)

    print("Computing histogram.")
    hist = compute_hist(L)

    print("Computing cdf.")
    cdf = compute_cdf(hist)

    print("Computing transformation.")
    transform = compute_transform(cdf, len(L))

    print("Setting transformed values to the image.")
    for y in range(image.shape[0]):
        currenty = y * imageLab.shape[1]
        for x in range(image.shape[1]):
            imageLab[y,x,0] = transform[L[x + currenty]]

    print("Moving image back to RGB.")
    return lab2rgb(imageLab) * 255

    # CUDA
    #  img_gpu = cuda.mem_alloc(img.nbytes)
    #  cuda.memcpy_htod(img_gpu, img)

    #  func = mod.get_function("equalize")
    #  func(img_gpu, block=(4,4,1))

    #  img_equalized = np.empty_like(img)
    #  cuda.memcpy_dtoh(img_equalized, img_gpu)




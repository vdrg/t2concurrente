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

__global__ void transform_values(float *img, int *values, float *transform, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
      int idx = x + y * width;
      img[3 * idx + 2] = transform[values[idx]];
    }
}

__global__ void transform_and_set(float *img, int *values, int bins, int width, int height)
{

    //extern __shared__ float transform[];

    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;

    //int idx = x + y * width;

    //if (idx < bins) {
    //  transform[idx] = (float) (cdf[idx] - cdf_min) / (width * height - 1);
   // }

    //__syncthreads();

    //if (x < width && y < height) {

     // img[3 * idx + 2] = transform[values[idx]];
   // }
}

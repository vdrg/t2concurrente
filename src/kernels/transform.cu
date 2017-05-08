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

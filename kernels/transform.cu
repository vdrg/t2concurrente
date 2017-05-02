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



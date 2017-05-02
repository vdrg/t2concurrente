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



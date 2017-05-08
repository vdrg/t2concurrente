__global__ void hist(int *L, int size,  int *hist, int n)
{
    int idx, offset;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    offset = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offset) {
        atomicAdd(&hist[L[i]],1);
    }
}

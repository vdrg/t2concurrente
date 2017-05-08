
__device__ void hsv_rgb_single(float h, float s, float v, unsigned char *r, unsigned char *g, unsigned char *b)
{

  // Adapted and simplified from https://github.com/jakebesworth/Simple-Color-Conversions

  /* Convert hue back to 0-6 space, floor */
  const float hex = h * 6;
  const unsigned char primary = (int) hex;
  const float secondary = hex - primary;

  float x = (1.0 - s) * v;
  float y = (1.0 - (s * secondary)) * v;
  float z = (1.0 - (s * (1.0 - secondary))) * v;

  float *rp, *gp, *bp;
  switch(primary) {
    case 0: rp = &v; gp = &z; bp = &x; break;
    case 1: rp = &y; gp = &v; bp = &x; break;
    case 2: rp = &x; gp = &v; bp = &z; break;
    case 3: rp = &x; gp = &y; bp = &v; break;
    case 4: rp = &z; gp = &x; bp = &v; break;
    case 5: 
    default: rp = &v; gp = &x; bp = &y; break;
  }

  *r = *rp * 255.0;
  *g = *gp * 255.0;
  *b = *bp * 255.0;
}

__global__ void hsv_rgb(float *img, unsigned char *result, int width, int height) 
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < width && y < height) {
    int idx = (x + y * width) * 3;
    hsv_rgb_single(img[idx], img[idx + 1], img[idx + 2], &result[idx], &result[idx + 1], &result[idx + 2]);
  }
}

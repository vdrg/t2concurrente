// Adapted from https://github.com/jakebesworth/Simple-Color-Conversions

__device__ void rgb_hsv_single(unsigned char rc, unsigned char gc, unsigned char bc, float *h, float *s, float *v)
{
  float min, max, delta;
  float r, g, b;


  r = (float) rc / 255.0;
  g = (float) gc / 255.0;
  b = (float) bc / 255.0;

  min = r < g ? r : g;
  min = min < b ? min : b;

  max = r > g ? r : g;
  max = max > b ? max : b;

  delta = max - min;

  *v = max;
  *s = max < 0.0001 ? 0 : delta / max;

  if(*s < 0.001) *h = 0;
  else if(r == max) *h = g == min ? 5 + (max - b) / delta : 1 - (max - g) / delta;
  else if(g == max) *h = b == min ? 1 + (max - r) / delta : 3 - (max - b) / delta;
  else if(b == max && r == min) *h = 3 + (max - g) / delta;
  else *h = 5 - (max - r) / delta;

  *h /= 6;
  *h = *h < 1 ? *h : 1;
}

__global__ void rgb_hsv(unsigned char *img, float *result, int width, int height) 
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < width && y < height) {
    int idx = (x + y * width) * 3;
    rgb_hsv_single(img[idx], img[idx + 1], img[idx + 2], &result[idx], &result[idx + 1], &result[idx + 2]);
  }
}

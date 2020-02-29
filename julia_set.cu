#include <cuda_runtime.h>
#include<iostream>
#include "cpu_bitmap.h"
using namespace std;
#include <device_launch_parameters.h>

#define DIM 1000

struct cuComplex {
	float r, i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) { return r * r + i * i; }

	//Operator overloading to define custom operations on struc when used 
	//Automatically called when used b/w struc
	__device__ cuComplex operator*(const cuComplex& a)
	{
		//Real comp multiply, imag comp become negative prod (j * j = (((sqrt(-1))^2))
		//Image comp get scaled by remaining real 
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};


//Ensure func ran on device 
__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	//Shift pixel coordinate to one located in complex space
	//Divide by DIM/2 to center complex plane about image, then offset with given x-coordinate, image must span range of [-1, 1], divide again by length: DIM/2
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	//Struct for forming complex numbers 
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;

	//Func called iteratively instead of recursively 
	for (i = 0; i < 200; i++)
	{
		//Julia set iteration formula
		a = a * a + c;

		//Square func inside of struct, used to iteratively expand at Julia set val, see if it bounds (converges)
		//If at any point it grows too much, obviously will not shrink after, assume failed case and continue in loop
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}


__global__ void kernel(unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	//Dim equal to image size, one block per pixel therefore can iterate via IDs
	//Define offset to incr ptr by, determine current location in grid by IDs and dimension, then per elem iterate through four floats at that point
	int offset = x + y * gridDim.x;

	int juliaVal = julia(x, y);
	//Each elem float, offset by four spaces at each ID then iterate through its bits 
	ptr[offset * 4 + 0] = 255 * juliaVal;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	//Create ptr to bitmap, then allocate GPU space for entire grid size (length * width) multiplied by four (store float at each elem)
	unsigned char *dev_bitmap;
	//Set block to image size for easy iteration later 
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	//dim3 special CUDA datatype: pass grid and block dimensions, all default to 1; essentially 3-elem vec
	//(DIM, DIM) = block of dim: DIM X DIM X 1
	dim3 grid(DIM, DIM);
	//Each point of Julia set check computed indepedent of another, pass 2D grid and kernel will create func copies equivalent to size
	kernel<<<grid, 1 >>>(dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}


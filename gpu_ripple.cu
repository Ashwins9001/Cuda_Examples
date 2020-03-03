#include <cuda_runtime.h>
#include<iostream>
#include "cpu_anim.h"
using namespace std;
#include <device_launch_parameters.h>
//Window dim
#define DIM 512 

//Create struct to hold device and host pointers to image bitmap
struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d)
{
	cudaFree(d->dev_bitmap);
}

void generate_frame(DataBlock *d, int render)
{
	//Create 2D grid of blocks, for every 16 threads there exists one
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel<<<blocks, threads >>>(d->dev_bitmap, render);
	//Copy contents at current render/frame, into dev bitmap for changing frame and computations 
	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

	
__global__ void kernel(unsigned char *ptr, int ticks)
{
	//Index one of the threads to an image pos
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	//Create varying grey vals depending on pixel 
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	//Offset into output buffer for window generation when ready 
	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

int main(void)
{
	DataBlock data;
	//construct with ref to data, shares same address, CPUAnimBitmap implements void ptr which implies that it can be associated w any data type
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	//Set up DataBlock ptr for host
	data.bitmap = &bitmap;
	cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size());
	//Notation below for function ptrs, pass address of each one for system to call back func when event occurs 
	//For rendering want to continuously createa and destroy frames to optimise GPU space and to upate image
	//First arg is return type, second is input params 
	bitmap.anim_and_exit((void(*)(void*, int))generate_frame, (void(*)(void*))cleanup);
}


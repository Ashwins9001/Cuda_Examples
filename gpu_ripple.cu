#include <cuda_runtime.h>
#include<iostream>
#include "cpu_anim.h"
using namespace std;
#include <device_launch_parameters.h>
#define DIM 1000

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


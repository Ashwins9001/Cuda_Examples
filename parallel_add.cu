#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>

//defn constant
#define N 10

__global__ void add(int* a, int* b, int* c)
{
	//CUDA consists of 2D blocks that contain threads which run simultaneously
	//Take id along x-dim for indexing
	int id = blockIdx.x;
	if (id < N)
	{
		c[id] = b[id] + a[id];
	}
}
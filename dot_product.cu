#include <cuda_runtime.h>
#include<iostream>
#include "cpu_bitmap.h"
using namespace std;
#include <device_launch_parameters.h>

/*
CUDA C has a __shared__ memory section where a copy of var is made for each block & threads within a block can all access var but cannot see or modify copy in other blocks
Must synchronize, if thread A writes to var and thread B wants to modify, must wait and ensure write is done else race condition occurs where correctness of var unknown
Addl shared mem buffers are physically on GPU, as opposed to off-chip DRAM which makes for much faster calls and reduced latency 
*/

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock]; //arr of caches equal to size of 256, each thread has spot to store temp vals & must wait for all writes to finish before another iteration where val retrieved for more modification 
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheID = threadIdx.x;
	float temp = 0;
	//iteratively take sum of products (dot prod) by creating threads for each arr elem
	//all threads will technically run at same hardware location, however abstract into larger set incase vector exceeds length
	//for small enough vals each thread theoretically computes single sum, yet again for large vecs, can continue greater iterations
	while (threadID < N)
	{
		temp += a[threadID] * b[threadID];
		threadID += blockDim.x * gridDim.x;
	}
	cache[cacheID] = temp; //shared mem buffer to store running sum per thread 
}
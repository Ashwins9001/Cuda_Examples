#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>

//defn constant, thread limit
#define N 65535


__global__ void add(int* a, int* b, int* c)
{
	//CUDA consists of 2D blocks that contain threads which run simultaneously
	//Take id along x-dim for indexing
	//Check if within bounds, update 
	int id = blockIdx.x;
	if (id < N)
	{
		c[id] = b[id] + a[id];
	}
}

int main(void) {
	int a[N], b[N], c[N];
	//Create ptr in host so they can pass by ref AND for GPU to have return addr when kernel execution completes 
	//At end, ptr contents are transferred 
	//Cannot access ptr from host as doing so may conflict with ongoing threads
	int *dev_a, *dev_b, *dev_c;

	//Allocate GPU memory
	//Don't want return type for mem pointer, its job is to simply set up resources
	//Address of pointer is what mem stores
	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);

	//Populate both local arrays 
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	//Copy local values over into dev_ptr and send to kernel
	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);

	//Invoke device, N within <<<>>> defines N blocks are allocated to run in parallel
	add <<<N, 1 >>> (dev_a, dev_b, dev_c);

	//Copy device values from dev_ptr back to local host
	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		cout << "Element at " << i << " is: " << c[i] << endl;
	}

	//Free GPU space 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}



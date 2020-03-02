#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>
#define N 5

__global__ void add(int* a, int* b, int* c)
{
	int id = threadIdx.x;
	if (id < N)
	{
		c[id] = b[id] + a[id];
	}
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);
	//Invoke device, N within <<<>>> defines N threads are allocated to run in parallel within one block 
	add <<<1, N>>> (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
	{
		cout << "Element at " << i << " is: " << c[i] << endl;
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}



#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>
#define N (1024 * 1024)

__global__ void add(int *a, int *b, int *c)
{
	//blockDim is num threads/block, multiplied by block number to index to one of them, then select thread inside block via thread Id
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	//Max 65 535 blocks, with 512 threads each ~ 8 million elements, if vector exceeds that amount require a soln
	//Run arbitrary number of blocks and threads
	//Done at each parallel process, allows a single launch of threads to iteratively cycle through all available indices of vector 
	//As long as each thread begins at a unique index-val, all will iterate arr without affecting one another  
	while (threadID < N)
	{
		c[threadID] = a[threadID] + b[threadID];
		//Add
		threadID += blockDim.x * gridDim.x;
	}
}

int main(void)
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

	//Could use add<<(N+127/128), 128>>>() to ensure blocks only created for largest multiples of 128, however this can create too many blocks if N >> 65 535 (limit)
	//To ensure that doesn't occur, use arbitrary num blocks
	add<<<128, 128 >>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if ((a[i] + b[i]) != c[i])
		{
			cout << "Error " << c[i] << " != " << a[i] << " + " << b[i] << endl;
			success = false;
		}
	}
	if (success) cout << "Last element is: " << c[N-1] << endl;

}
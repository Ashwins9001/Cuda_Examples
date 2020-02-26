#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>

//programs ran on GPU, called device
//func itself is a kernel
//global identifier indicates func ran on device, not host
//main code -> compiled via host, kernel code -> compiled via device
__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

//programs run on CPU, called host
int main(void) {

	//invoke code from host
	//angled brackets indicate args to runtime, influence how code executes on GPU
	//local host int c
	int c;

	//create device ptr which creates mem via cudaMalloc
	int *dev_c;

	//allocate device memory via cudaMalloc, first arg = ptr to ptr that holds new mem addr, second arg = size mem
	//cast as addr of pointer to int c as void double ptr
	//handle error micro to return error code if exists
	//cudaMalloc returns ptr to mem, cannot use this ptr to read from or write to memory from code that executes on host
	//CAN pass mem ptr to func exec on device, or to read/write to device, and to func that execute on host
	//important error, C compiler won't notify
	//cannot deref dev_c
	cudaMalloc ( (void**)&dev_c, sizeof(int) );

	//run kernel call, pass ptr for device to use
	add<<<1,1>>>(2, 7, dev_c);

	//copy block of memory at address of ptr c inside device, to local int c inside host
	//number of bytes copied indicated by sizeof(int), and define type of copy is to host
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("2 + 7 = %d\n", c);
	cudaFree(dev_c);
	return 0;
}

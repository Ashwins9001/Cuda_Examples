#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#include <device_launch_parameters.h>

int main(void) {
	//struct containing info such as name, threads/block, etc.
	cudaDeviceProp devProp;
	int count;
	//pass addr of var, get method populates 
	cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&devProp, i);
		cout << "Name: " << devProp.name << endl;
		cout << "Clock rate: " << devProp.clockRate << endl;
		cout << "Total global memory: " << devProp.totalGlobalMem << endl;
		printf("Max thread dimensions: (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
	}
}
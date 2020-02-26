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

	//Check for GPUs capable of double-precision floating-point math, only available on those w computer capability >= 1.3
	//Fill block of mem w particular struct that is threshold, then use cudaChooseDevice to find one with best criteria
	//Lower overhead, instead of iterating through all in for loop
	int deviceID;
	cudaGetDevice(&deviceID);

	//copy obj for first 0 iterations and allocate mem 
	memset(&devProp, 0, sizeof(cudaDeviceProp));
	devProp.major = 1;
	devProp.minor = 3;
	cudaChooseDevice(&deviceID, &devProp);
	cout << "Device ID with closest capability: " << deviceID << endl;

	//host comms w this device now
	cudaSetDevice(deviceID);
}
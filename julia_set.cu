#include <cuda_runtime.h>
#include<iostream>
#include "cpu_bitmap.h"
using namespace std;
#include <device_launch_parameters.h>

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();
	kernel(ptr);
	bitmap.display_and_exit();
}
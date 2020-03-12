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


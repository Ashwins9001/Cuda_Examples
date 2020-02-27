#include <cuda_runtime.h>
#include<iostream>
#include "cpu_bitmap.h"
using namespace std;
#include <device_launch_parameters.h>

#define DIM 1000

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	//get current location on bitmap and pass to kernel
	unsigned char *ptr = bitmap.get_ptr();



	kernel(ptr);
	bitmap.display_and_exit();
}

void kernel(unsigned char* ptr) {
	for (int y = 0; y < DIM; y++)
	{
		for (int x = 0; x < DIM; x++)
		{
			int offset = x + y * DIM;
			int juliaVal = julia(x, y);
			//increment by blocks of four each time and fill adj bitmap cells
			//call Julia on each to determine if point is within set or not 
			//Remember for each point of Julia set, it gets iterated and check whether it grows toward infinity or converges 
			//Set consist of all converging points and outer edges form fractal 
			//RGB val, if juliaVal = 1: red, else black
			ptr[offset * 4 + 0] = 255 * juliaValue;
			ptr[offset * 4 + 1] = 0;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
	}
}

int julia(int x, int y)
{
	const float scale = 1.5;
	//Shift pixel coordinate to one located in complex space
	//Divide by DIM/2 to center complex plane about image, then offset with given x-coordinate, image must span range of [-1, 1], divide again by length: DIM/2
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	//Struct for forming complex numbers 
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;

	//Func called iteratively instead of recursively 
	for (i = 0; i < 200; i++)
	{
		//Julia set iteration formula
		a = a * a + c;

		//Square func inside of struct, used to iteratively expand at Julia set val, see if it bounds (converges)
		//If at any point it grows too much, obviously will not shrink after, assume failed case and continue in loop
		if (a.magnitude2() > 1000)
			return 0;
	}
	return -1;

}

struct cuComplex {
	float r, i;
	cuComplex(float a, float b) : r(a), i(b) {}
	float magnitude2(void) { return r * r + i * i; }

	//Operator overloading to define custom operations on struc when used 
	//Automatically called when used b/w struc
	cuComplex operator*(const cuComplex& a)
	{
		//Real comp multiply, imag comp become negative prod (j * j = (((sqrt(-1))^2))
		//Image comp get scaled by remaining real 
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}

	cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};
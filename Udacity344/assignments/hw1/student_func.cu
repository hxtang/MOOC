#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
	int pixel_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (pixel_id < numRows * numCols) 
	{
		uchar4 rgbw_in = rgbaImage[pixel_id];
		char gray_out = .299f * rgbw_in.x + .587f * rgbw_in.y + .114f * rgbw_in.z;
		greyImage[pixel_id] = gray_out;
	}
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 gridSize(1024);
  const dim3 blockSize((numRows*numCols-1)/gridSize.x+1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}
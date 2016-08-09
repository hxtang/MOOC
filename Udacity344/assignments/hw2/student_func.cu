#include "reference_calc.cpp"
#include "utils.h"

//GPU global variables local to this file
unsigned char *d_red, *d_green, *d_blue; //single channel of input image on GPU global memory
float         *d_filter; // filter on GPU global memory

// allocate space for GPU global variables local to this file
void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                        const float* const h_filter, const size_t filterWidth)
{

	//allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

	//allocate memory for the filter
	checkCudaErrors(cudaMalloc(&d_filter,   sizeof(float) * filterWidth));

	float flt[16] = {0};
	for (int k = 0; k < filterWidth; ++k)
	for (int j = 0; j < filterWidth; ++j) flt[k] += h_filter[k*filterWidth+j];

	checkCudaErrors(cudaMemcpy(d_filter, flt, sizeof(float) * filterWidth, cudaMemcpyHostToDevice));
}

// release space for GPU global variables local to this file
void cleanup() 
{
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter));  
}

__global__
void gaussian_blur_h(unsigned char* const inputChannel,
           int numRows, int numCols,
           const float* const filter, const int filterWidth)
{

	int y = blockIdx.x;
	int x = threadIdx.x;

	__shared__ float flt[16], input[2048];

	if (x < filterWidth)
	flt[x] = filter[x];
	__syncthreads();

	input[x] = inputChannel[y*numCols + x];
	__syncthreads();

	const int FH = filterWidth/2;
	float sum = 0;
	for (int fc = -FH; fc <= FH; ++fc)
	{
		int c = min(max(x + fc, 0), numCols-1);
		sum += flt[FH+fc] * input[c];			
	}

	inputChannel[y*numCols + x] = sum;
}


__global__
void gaussian_blur_v(const unsigned char* const inputChannel,
					 unsigned char* const outputChannel,
					 int numRows, int numCols,
					 const float* const filter, const int filterWidth)
{
	__shared__ float flt[16], input[2048];

	if (threadIdx.x < filterWidth)
	flt[threadIdx.x] = filter[threadIdx.x];
	__syncthreads();

	int x = blockIdx.x;
	int y = threadIdx.x;
	input[y] = inputChannel[y*numCols + x];
	__syncthreads();

	const int FH = filterWidth/2;
	float sum = 0;
	for (int fr = -FH; fr <= FH; ++fr) {
		int r = min(max(y + fr, 0), numRows-1);
		sum +=  input[r]*flt[FH+fr];			
	}

	outputChannel[y*numCols + x] = sum;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
              int numRows,
              int numCols,
              unsigned char* const redChannel,
              unsigned char* const greenChannel,
              unsigned char* const blueChannel)
{
	const int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_1D_pos >= numCols * numRows)   return;

	uchar4 rgba = inputImageRGBA[thread_1D_pos];
	redChannel[thread_1D_pos] = rgba.x;
	greenChannel[thread_1D_pos] = rgba.y;
	blueChannel[thread_1D_pos] = rgba.z;  
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
               const unsigned char* const greenChannel,
               const unsigned char* const blueChannel,
               uchar4* const outputImageRGBA,
               int numRows,
               int numCols)
{
	const int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_1D_pos >= numCols * numRows)   return;

	unsigned char red   = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue  = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

//	Applies gaussian blur kernel to image stored in d_inputImageRGBA
//  h_inputImageRGBA	input image in CPU in RGBA
//  d_inputImageRGBA	input image in GPU global memory in RGBA (allocated/memcpyed/released externally)
//  d_outputImageRGBA	input image in GPU global memory in RGBA (allocated/released externally)
//  numRows				image height
//  numCols				image width
//  d_redBlurred		output image red channel  (allocated/released externally)
//  d_greenBlurred		output image green channel(allocated/released externally)
//  d_blueBlurred		output image blue channel (allocated/released externally)
void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                unsigned char *d_redBlurred, unsigned char *d_greenBlurred, unsigned char *d_blueBlurred,
                const int filterWidth)
{	
	int numPix = numRows * numCols;
	const dim3 gridSize(1024);
	const dim3 blockSize((numCols*numRows-1)/1024+1);

	//Launch a kernel for separating the RGBA image into different color channels
	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols,
	                                    d_red, d_green, d_blue);

	// Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.

	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int sharedSize = filterWidth*filterWidth*sizeof(float);

	gaussian_blur_h<<<numRows, numCols>>>(d_red,   numRows, numCols, d_filter, filterWidth);
	gaussian_blur_h<<<numRows, numCols>>>(d_green, numRows, numCols, d_filter, filterWidth);
	gaussian_blur_h<<<numRows, numCols>>>(d_blue,  numRows, numCols, d_filter, filterWidth);

	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	gaussian_blur_v<<<numCols, numRows>>>(d_red,   d_redBlurred,   numRows, numCols, d_filter, filterWidth);
	gaussian_blur_v<<<numCols, numRows>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blur_v<<<numCols, numRows>>>(d_blue,  d_blueBlurred,  numRows, numCols, d_filter, filterWidth);

	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Now we recombine your results. We take care of launching this kernel for you.
	recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
	                                     d_outputImageRGBA, numRows, numCols);
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

#include "reference_calc.cpp"
#include "utils.h"
#include "stdio.h"

__global__ void compMin(float *d_out, const float *d_in, int len)
{
	extern __shared__ float s_cache[];
	
	// load data into local memory
	int tid = threadIdx.x, bid = blockIdx.x, id = bid * blockDim.x + tid;
	if (id < len) s_cache[tid] = d_in[id]; else s_cache[tid] = 275;
	__syncthreads();
	
	// compute local min
	for (int l = blockDim.x/2; l > 0; l >>= 1) {
		if (tid < l) s_cache[tid] = min(s_cache[tid], s_cache[tid+l]); 
    	__syncthreads();
	}
	if (tid == 0) d_out[bid] = s_cache[0];
}

__global__ void compMax(float *d_out, const float *d_in, int len)
{
	extern __shared__ float s_cache[];
	
	// load data into local memory
	int tid = threadIdx.x, bid = blockIdx.x, id = bid * blockDim.x + tid;
	if (id < len) s_cache[tid] = d_in[id]; else s_cache[tid] = 0;
	__syncthreads();
	
	// compute local min
	for (int l = blockDim.x/2; l > 0; l >>= 1) {
		if (tid < l) s_cache[tid] = max(s_cache[tid], s_cache[tid+l]); 
    	__syncthreads();
	}
	if (tid == 0) d_out[bid] = s_cache[0];
}

__global__ void compPdf(unsigned int *d_out, const float *d_in, 
                        float lumMin, float lumRange, int numPixels, int numBins) 
{

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numBins) d_out[id] = 0;
	
	if (id >= numPixels) return;
	int bin_id = (d_in[id] - lumMin) * numBins/lumRange;
	bin_id = min(bin_id, numBins-1);
	atomicAdd(&d_out[bin_id], 1);
}

__global__ void compCdf(unsigned int *d_out, unsigned int *d_in, int numBins) 
{
	extern __shared__ float s_cache[];
	
	int tid = threadIdx.x;
	if (blockIdx.x >0 || tid < numBins) s_cache[tid] = d_in[tid]; else return;
	
	for (int l = 1; l < numBins; l <<= 1) {
		if (tid >= l) s_cache[tid+numBins] = s_cache[tid-l] + s_cache[tid];
		else s_cache[tid+numBins] = s_cache[tid];
		__syncthreads();
		s_cache[tid] = s_cache[tid+numBins];
		__syncthreads();
	}
		
	d_out[tid] = tid == 0? 0 : s_cache[tid-1];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    size_t numPixels = numCols * numRows;
    const int BLOCKS = 1024, THREADS = (numPixels-1)/BLOCKS+1;
    
    float *d_cache;
    cudaMalloc((void**) &d_cache, BLOCKS*sizeof(float));

    compMin<<<BLOCKS, THREADS, THREADS*sizeof(float)>>>(d_cache, d_logLuminance, numPixels);
    compMin<<<1, BLOCKS, BLOCKS*sizeof(float)>>>(d_cache, d_cache, BLOCKS);
    cudaMemcpy(&min_logLum, d_cache, sizeof(float), cudaMemcpyDeviceToHost);
    
    compMax<<<BLOCKS, THREADS, THREADS*sizeof(float)>>>(d_cache, d_logLuminance, numPixels);
    compMax<<<1, BLOCKS, BLOCKS*sizeof(float)>>>(d_cache, d_cache, BLOCKS);
    cudaMemcpy(&max_logLum, d_cache, sizeof(float), cudaMemcpyDeviceToHost);

    compPdf<<<BLOCKS, THREADS>>>(d_cdf, d_logLuminance, min_logLum, max_logLum-min_logLum, numPixels, numBins);
    
    compCdf<<<1, numBins, numBins*2*sizeof(unsigned int)>>>(d_cdf, d_cdf, numBins);
    
    cudaFree(d_cache);
}
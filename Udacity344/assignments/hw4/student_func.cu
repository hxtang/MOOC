#include "reference_calc.cpp"
#include "utils.h"

__global__ void predicator(unsigned int* const d_output, unsigned int* d_inputV, size_t len, int bit_id) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= len) return;
    d_output[id] = ((d_inputV[id] >> bit_id) & 1);
    assert(d_output[id] ==0 || d_output[id]==1);
}

__global__ void flip(unsigned int *d_input, int len) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= len) return;
    d_input[id] = 1-d_input[id];
    assert(d_input[id]==0 || d_input[id] ==1);
}

__global__ void scanEx(unsigned int* const d_output, unsigned int* d_input, unsigned int* d_offset, int len) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x, bid = blockIdx.x, N = blockDim.x*2, base = bid*N;
	
	// load input to sdata
    int id1 = tid*2   + base;	
    int id2 = tid*2+1 + base;	
	sdata[tid*2]   = id1 < len? d_input[id1]:0;
    sdata[tid*2+1] = id2 < len? d_input[id2]:0; 
	__syncthreads();
	
    int offset = 1, last = sdata[N-1];
    
    // down scan
	for (int d = N/2; d > 0; d = d/2) {
		if (tid < d) {
			int i1 = offset*(2*tid+1)-1;
			int i2 = offset*(2*tid+2)-1;
			sdata[i2] += sdata[i1];
		}
		offset *= 2;		
		__syncthreads();
	}
	
	if (tid == 0) sdata[N-1] = 0;
	__syncthreads();	

    //up scan
	for (int d = 1; d<=N/2; d = d*2) {
		offset /= 2;

		if (tid < d) {
			int i1 = offset*(2*tid+1)-1;
			int i2 = offset*(2*tid+2)-1;
			unsigned int tmp = sdata[i1];
			sdata[i1] = sdata[i2];
			sdata[i2] += tmp;
		}
		__syncthreads();
	}

    // output result
	if (id1<len) d_output[id1] = sdata[tid*2];
	if (id2<len) d_output[id2] = sdata[tid*2+1];
	
	if (tid == 0) d_offset[bid] = sdata[N-1] + last;

}

//compute global indices and scatter results
__global__
void merge(  unsigned int* const d_outputV, unsigned int* const d_outputP,
             unsigned int* const d_inputV,  unsigned int* const d_inputP,
			 unsigned int* const d_pred,
             unsigned int* const d_index0, unsigned int* const d_index1,
			 unsigned int* const d_offset0, unsigned int* const d_offset1,
			 unsigned int* const d_count, unsigned int len) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= len) return;
    
	int id0 = d_index0[id] + d_offset0[blockIdx.x];
	int id1 = d_index1[id] + d_count[0] + d_offset1[blockIdx.x];
	int p = d_pred[id];
    int index = p==0? id0:id1;

	d_outputV[index] = d_inputV[id];
	d_outputP[index] = d_inputP[id];	
}

inline unsigned int next_power_of_two(unsigned int v) {
    --v;
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    return v+1;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const int BSZ = 256, GSZ = (numElems-1)/BSZ+1;
	const int HBSZ = BSZ/2, PGSZ = next_power_of_two(GSZ), HGSZ = PGSZ/2;
    
    unsigned int* d_pred, *d_index0, *d_index1, *d_offset0, *d_offset1, *d_count;
    unsigned int* d_interVals, *d_interPos;
    
    cudaMalloc((void **)&d_pred,      sizeof(unsigned int) * numElems);
    cudaMalloc((void **)&d_index0,    sizeof(unsigned int) * numElems);
    cudaMalloc((void **)&d_index1,    sizeof(unsigned int) * numElems);
    cudaMalloc((void **)&d_offset0,   sizeof(unsigned int) * GSZ);
    cudaMalloc((void **)&d_offset1,   sizeof(unsigned int) * GSZ);
    cudaMalloc((void **)&d_count,     sizeof(unsigned int) * 1);	
    cudaMalloc((void **)&d_interVals, sizeof(unsigned int) * numElems);
    cudaMalloc((void **)&d_interPos,  sizeof(unsigned int) * numElems);
 
    cudaMemcpy(d_interVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_interPos,  d_inputPos,  numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    for (int k = 0; k < 8 * sizeof(unsigned int); ++k) {
		predicator<<<GSZ, BSZ>>>(d_pred, d_interVals, numElems, k);

		//compute local indices
		scanEx<<<GSZ, HBSZ, BSZ*sizeof(unsigned int)>>>(d_index1, d_pred, d_offset1, numElems);
		scanEx<<<1, HGSZ, PGSZ*sizeof(unsigned int)>>>(d_offset1, d_offset1, d_count, GSZ);
		flip<<<GSZ, BSZ>>>(d_pred, numElems);

		scanEx<<<GSZ, HBSZ, BSZ*sizeof(unsigned int)>>>(d_index0, d_pred, d_offset0, numElems);
		scanEx<<<1, HGSZ, PGSZ*sizeof(unsigned int)>>>(d_offset0, d_offset0, d_count, GSZ);
        flip<<<GSZ, BSZ>>>(d_pred, numElems);
        
        merge<<<GSZ, BSZ>>>(d_outputVals, d_outputPos, d_interVals, d_interPos,
							d_pred, d_index0, d_index1, d_offset0, d_offset1, d_count, 
							numElems);

        //swap input and output
        cudaMemcpy(d_interVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_interPos,  d_outputPos,  numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        
    }
    
    cudaFree(d_pred);
    cudaFree(d_index0);
    cudaFree(d_index1);
    cudaFree(d_interVals);
    cudaFree(d_interPos);
    cudaFree(d_offset0);
    cudaFree(d_offset1);
    cudaFree(d_count);
}
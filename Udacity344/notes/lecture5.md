# Lecture 5  Optimizing GPU programs

###Overview

**Weak v.s. Strong Scaling**

* Weak: solve a bigger problem, or more small ones (solution size varies with fixed problem size per core)
* Strong: run a problem faster (solution size varies with fixed total problem size)

**Levels of Optimization**

* Pick good algorithms: choose algorithms fundamentally parallel
* Basic principles for efficiency
* Arch-specific detailed optimization: avoid bank conflicts & optimizing registers
* Micro-optimization: float point denormalization hacks

**Principles of GPU programs**

* Increase arithmetic intensity (decrease time spent on memory ops)
* Coalesce global memory accesses
* Avoid thread divergence

**System optimization: APOD**

* Analyze: profile entire application -- where can it benefit and by how much?
* Parallize: pick an approach (libraries, Directives such as OpenMP/OpenACC, programming language) or pick an algorithm
* Optimize: profile-driven optimization
* Deploy: "make it real", "small" speedups help

###Analysis
**Profiling parallizable time**

* Use profilers: Gprof, VTune, VerySleepy
* Parallize "hotspots": (Amdahl's law) assume p to be the % of parallizable time, max speed up is 1/(1-p)
* Refactor

**DRAM utiliziation**

* Theoretical peak bandwidth: memory block * memory bus size
* In practice, 60%-75% is good, 40%-60% is ok
* Measurements: nSight, nvvp
* Note coalesing

**Parallize**

* Max parallelism not always best performance ("granularity coarsening")
* Considerations: usage of memory and thread (there's a limit for each block/processor)

**Tiling**

e.g. matrix transpose:

```c
const int N= 1024;	// matrix size will be NxN
const int K= 16;	// tile size will be KxK

__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
    __shared__ float sdata[K][K];
	
    int i, j;
    i = threadIdx.x + blockIdx.x * K;
    j = threadIdx.y + blockIdx.y * K;
    sdata[threadIdx.y][threadIdx.x] = in[i + j*N];
    __syncthreads();
    
    i = threadIdx.x + blockIdx.y * K;
    j = threadIdx.y + blockIdx.x * K;
    out[i + j*N] = sdata[threadIdx.x][threadIdx.y];
}
```

###Parallelize
**Little's law**

* Goal: max bandwidth
* The law: Useful bytes delivered = average latency * bandwidth
* Solutions: reduce latency or reduce #thread per block to increase #block per SM)

**Occupancy**

Each SM has a limited number of 

* thread blocks (8)
* Threads (1536/2049)
* Registers for all threads (65536)
* Bytes of shared memory (16-48K)

How to affect occupancy

* Control #shared memory (e.g. tile size)
* Change #thread, #blocks in kernel launch
* Compilation options to control register usage
* There's a balance between occupancy / speed of per thread

###Optimization
**maximize useful computation/second**

* Minimize time waiting at barriers
* Minimize thread divergence: 
  * avoid branchy code, esp; 
  * avoid adjacent threads to take different paths; 
  * avoid large imbalance in thread workloads (loops, recursive calls)

**Warp:** set of threads that executes the same instruction at a time

* CPU level: SIMD(Single Instruction Multiple Data), e.g. SSE, AVX vector register
* GPU level: SIMT(Single Instruction Multiple Threads)
* Each warp contains a max of 32 threads

**Consider fast math**

* Use double precision only when you mean it (fp64>fp32)
* Do a = b + 1.2f; rather than a  = b + 1.2;
* Use intrinsics/compile flags of fast math when possible

**Host-GPU interaction**

* Pinned(page locked) host memory
* When doing memcpy: staging(cudaHostRegister) occurs before tranmitting to GPU via PCI express bus
* Enables asynchronized transfers(doesn't block transfer in other streams before a tranfer finishes)

**Streams: sequence of ops that execute in order**

* Overlap memory & compute
* Help fill GPUs with small kernels (many small tasks hard to parallelize, or computation with narrow phases, e.g. reduce)

Example:
```c
cudaStream_t s1;
cudaStreamCreate(&s1);
…
cudaMemcpyAsync(...., s1);
A<<< >>>(..., s2);
cudaMemcpyAsync(..., s3);
A<<< >>>(.., s4);
…
cudaStreamDestroy(..., s1);
```

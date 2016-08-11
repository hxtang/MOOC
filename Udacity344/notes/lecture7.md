# Lecture 7 Advanced topics

##Seven parallel optimization patterns
###**Data layout transformation**

Global memory coalesing is importance because DRAM systems transfer large chunks of data per transaction

**Methods:**

* "Burst utilization". E.g. array of structure -> structure of array
* Array of structure of tiled arrays (ASTA) to address "Partition camping" etal
  
###**Scatter-to-gather transformation**

Gather runs more efficiently than scatter since there's no competition in writing a value

###**Tiling**

Buffer data into fast on-chip storage (shared memory) for repeat access

###**Privatization**

Threads sharing input is good;
Threads sharing output is bad

###**Binning/spatial data structures**

Build data structure that maps output locations to (a small subset of ) the relevant input data

e.g. compute "accessible population" of a city

* Overlay a grid on the map
* For each city, find cities in the grid containing the city and neighboring grids

###**Compaction**

Reduce output size to reduce work afterwards

###**Regularization (Load balancing)**

Reorganize input data to reduce load imbalance

Useful when 

* load is mostly evenly distributed, with outliers 
* application can predict load imbalance at runtime

##Resources
###Libraries

* Substitute library calls, e.g. saxpy ->cublasSaxpy
* Manage data locality
* Rebuild and link

**Numerical, image processing:**cuBlas/cuFFT/cuSparse/cuRand/NPP/Magma/CULA/ArrayFile

**Thrust**: analogous to C++ STL, host side interface, no kernels

* host code using thrust::device_vector
* In cpu code, call thurst::xxx
* Sort, scan, reduce, reduce-by-key
* Transform input vector
* Interoperate with CUDA code

Avoid boilerplate, exploits high performance code

**CUB: software reuse in CUDA kernels**

* Compile time binding with templates, typing, high performance
* Enables optimization, autotuning

Why CUB: accesing global memory can be complicated

* Loading halo regions
* Coalescing
* Little's law
* Kepler LDG intrinsics
Explicit shared memory achieves predictable high performance at the cost of burden on programmer

**CUB DMA:** template library to help use shared memory at high performance

CudaDMA objects: each shared memory buffer

* Explict transfer patterns: sequential, strided, indirect
* Programmability, portability, high performance

###Languages

* PyCUDA: python wrap of CUDA
* Copperhead: data-parallel subset of python
* CudaFortran: Fortran with CUDA constructs
* Halide: imaging processing DSL
* Matlab CUDA toolbox

###Crossplatform solutions
* OpenCL
* OpenGL compute
* OpenACC: Directives, Evolution of openMP

##Dynamic parallelism
Allow kernel to launch kernel

###Parallelisms
* Bulk parallelism: no dependencies among kernels
* Nested parallelism: call parallel functions sequentially inside a parallel function
* Task paralleism: call different tasks in parallel inside a parallel function
* Recursive parallelism: e.g. quicksort

###Programming model
just the same as launching kernel on CPU

###Things to watch out for
* Every thread executes the same program (lots of launches) - may want to launch only in one of the threads
* Each block executes independently - shared memory are private to locks
* A Block's private data is private - stream/events are private to blocks and cannot be passed to children

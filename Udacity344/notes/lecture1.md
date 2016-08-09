# Lecture 1    The GPU programming model

### GPU trades simple control for power-efficiency / throughput

* Consequence of simple ctrl: restrictive programming model

### CUDA programming model

![](https://github.com/hxtang/MOOC/blob/Udacity344/Udacity344/notes/images/Lec1_programming_model.png)

* CPU initiates all requests, GPU only does the job

* Maximize amount of computation\/communication


### A "hello world" example

**compile：**nvcc -o cube cube.cu

**CPU part:** int main\(int argc, char \*\* argv\)

* Declare GPU pointer & allocate memory \(resides in GPU global mem\)

* Transport CPU data to GPU

* Launch GPU kernel

* Copy from GPU to CPU


* Free space

**GPU part: **\_\_global\_\_ void cube\(float \* d\_out, float \* d\_in\)

* Use ThreadIdx, BlockDim, BlockIdx, to determine what to compute

* Do the work for a single thread \(may declare temp libray\)


```c
#include <stdio.h>

__global__ void cube(float * d_out, float * d_in) {
    int i = threadIdx.x;
    d_out[i] = d_in[i] * d_in[i] * d_in[i];
}

int main(int argc, char ** argv) {
    const int ARRAY_SIZE = 96; 
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float); 

    // generate the input array on the host float h_in[ARRAY_SIZE]; 
    for (int i = 0; i < ARRAY_SIZE; i++) { 
        h_in[i] = float(i); 
    } 
    float h_out[ARRAY_SIZE]; 

    // declare GPU memory pointers 
    float * d_in; 
    float * d_out; 

    // allocate GPU global memory cudaMalloc((void**) &d_in, ARRAY_BYTES); 
    cudaMalloc((void**) &d_out, ARRAY_BYTES); 

    // transfer the array to the GPU 
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the kernel 
    cube<<<1, ARRAY_SIZE>>>(d_out, d_in); 

    // copy back the result array to the CPU 
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost); 

    // print out the resulting array 
    for (int i =0; i < ARRAY_SIZE; i++) { 
        printf("%f", h_out[i]); printf(((i % 4) != 3) ? "\t" : "\n"); 
    } 

    // free GPU global memory
    cudaFree(d_in); 
    cudaFree(d_out); 
    return 0;
}
```

### Functions

* Allocates sz bytes starting \*ptr

  ```c
  cudaMalloc(void** ptr, size_t sz); 
  ```

* Copy sz bytes from src to dst

  ```c
  cudaMemcpy(void* dst, const void* src, size_t sz, enum cudaMemcpyKind kind);
  ```

  * transfer kind is in the form cudaMemcpy&lt;Src&gt;To&lt;Dst&gt; 
  * Src and Dst can be: Device or Host

* Launch kernels

  ```c
  __global__ kfun(<param_list>) {}; //kernel function
  dim3 GRID_SZ(gx, gy, gz);
  dim3 BLOCK_SZ(bx, by, bz);
  kfun<<<dim3 GRID_SZ, dim3 BLOCK_SZ>>> (<param_list>);
  ```

* grid size \(number of blocks\): \(gx, gy, gz\)

  * block size\(number of threads\/block\): \(bx, by, bz\)

  * parameter list: param\_list


* free space starting at ptr

  ```c
  cudaFree(void* ptr)
  ```



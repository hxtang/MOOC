#Lecture 2 GPU Hardware

###Parallel communication patterns

* Map: 1 -> 1

* Gather: N -> 1

* Scatter: 1->N

* Stencil: gather with fixed neighborhood

* Transpose: Array of Structure (AoS) -> Structure of Array (SoA)

* Reduce: all->1

* Scan/Sort: all->all

###GPU hardware


![](https://github.com/hxtang/MOOC/blob/Udacity344/Udacity344/notes/images/Lec2_GPU_Hardware.png)

* GPU: a bunch of streaming(SM) processors

* SM <-> thread block (cuda makes little assumption about where or when to run TB)

CUDA guarantees

* All threads of a block run in the same SM at the same time

* All blocks in the kernel finish before any blocks from the next kernel launches

Pros and Cons of CUDA

* Pro: flexible, therefore efficient and scalable

* Con: 

  * no assumption about block-to-SM assignment 

  * no communication among blocks

###Memory model

![](https://github.com/hxtang/MOOC/blob/Udacity344/Udacity344/notes/images/Lec2_GPU_Hardware.png)

* global memory is shared by all blocks

* _shared memory_ is shared by threads of the same block (declared in kernel function)

```c

__share__ int arr[128]; //declare shared memory of fixed size in kernel function

```

* lobal memory are local to threads(temps of kernel function)Speed: local >shared >>global

###Synchronization 

**barriers**

```c

__syncthreads();

```

**Atomic ops:**

```c

atomicAdd(); 

atomicCAS(); 

atomicMin(); 

atomicXOR();

```

Limitations:

* Data type limited (although other datatypes may be supported by implementing with atomicCAS).

* Still no order constraint

* Serializes access to memory

###Writing efficient programs

* maximize arithemetic intensity: #compute ops per thread / time spent on memory

* move frequently accessed data to fast memory

* Use coalessed global memory (e.g. manipulate on SoA before processing)

* avoid thread divergence


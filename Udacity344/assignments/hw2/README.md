#Solution to problem 2: Gaussian blur

###Problem
The program applies Gaussian blur to images.
This file implements the core part of the blur assuming blur kernel is 9x9.

###Key ideas
* Replaced the 2D Gaussian blur with two 1D Gaussian blur using separability of Gaussian blur
* Used shared memory, rather than global GPU memory
* Turned off cudaDeviceSynchronize() and checkCudaErrors() in the final submission
* Tuned grid size and block size to achieve best possible performance

###Spaces of improvements
This implementation runs <0.6ms on the course testing platform, the best performance reported online is about ~0.3ms, so there is still much space to improve.

* Change shared memory allocation from static to dynamic
* Test if performance can get better by direcly processing on AoS input
* Try rotate the images before/after per-column blurring to avoid coalescing
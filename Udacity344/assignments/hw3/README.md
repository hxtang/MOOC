#Solution to problem 3: HDR

###Problem
The program turns a linear RGB image into a high dynamic range image.
This file implements the core part of brightness histogram computation.

###Key ideas
* kernel funcitons compMin and compMax implement reduce algorithm
* kernel function compPdf implements simple histograming by atomic add
* kernel function compCdf implements Hills and Steels scan algorithm

###Spaces of improvement
The code runs ~0.2ms on the course testing platform. 

* try Blellock for compCdf
* try thread privatalization idea for histogram algorithm 
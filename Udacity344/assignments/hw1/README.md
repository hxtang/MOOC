#Solution to problem 1: RGB-to-Gray

###Problem
The program turns RGB images into a gray image.
This file implements the core part of RGB-to-gray conversion.

###Key ideas
* Treated the 2D problem as a 1D problem
* Avoided coalesing when defining the (block id, thread id) -> pixel mapping
* Checked if mapped pixel index is out of range
* Tuned grid size and block size for best performance

###Performance
The code runs <0.03ms on the course testing platform. Due to simplicity of the problem, it is hard to imagine there is much space to further optimize.
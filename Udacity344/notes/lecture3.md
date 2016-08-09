# Lecture 3 Reduce, Scan and Histogram

### Work Complexity v.s. Step Complexity:

* Work complexity: total amount of work

* Step complexity: amount of time to complete work in parallel

We can an algorithm **work efficient** if work complexity is asyntotically the same as sequential algorithm

### Reduce

* Input: a set of numbers $$a_0, a_1, a_2, a_3, ... a_{n-1}$$ and a binary associative operator $$\circ$$

* Output: $$a_0 \circ a_1 \circ a_2 \circ ... \circ a_{n-1}$$

**Algorithm:** iteratively reduces input array by half by applying op to pairwise elements

**Implementation:** two rounds of reduce

1. reduce within per block

2. reduce with a single block

###Scan

Input: a set of numbers $$a_0, a_1, a_2, a_3, ... a_{n-1}$$, a binary associative operator $$\circ$$, identity element $$I$$

Output: 

* Exclusive: $$I$$, $$a_0$$, $$a_0 \circ a_1$$, ... , $$a_0 \circ a_1 \circ a_2\circ ... \circ a_{n-2}$$

* Inclusive: $$a_0$$, $$a_0 \circ a_1$$, ... , $$a_0 \circ a_1 \circ a_2\circ ... \circ a_{n-1}$$

**Algorithms:**

* **Hills & Steels:** O(nlogn) work, logn steps, good when processors much more than work
* **Blellock:** O(n) work, 2logn steps, good when work much more than processors
![](https://github.com/hxtang/MOOC/blob/Udacity344/Udacity344/notes/images/Lec3_Hills_Steels.png "Hills and Steels")
![](https://github.com/hxtang/MOOC/blob/Udacity344/Udacity344/notes/images/Lec3_Blellock.png "Blellock")

###Histogram

* Method 1: atomic_add

* Method 2: Per-thread privatized (local) histogram, then reduce

* Method 3: Sort, then reduce by key

###Dynamic shared memory

In CPU code:

```c

kernel_fun<<<GRID_SIZE, BLOCK_SIZE, SHARED_MEM_SIZE_IN_BYTE>>(<param list>)

```

In GPU code:

```c

extern BYTE sh_data[];

```


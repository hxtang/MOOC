#Lecture 6 Applications of parallel computing patterns

###All pairs N-Body
* compute force between each pair of elements, and apply all the force on each body to move it
* Work complexity: N2

**Tiling: share global mem**

* Larger tile size -> less total main memory bandwidth (good, but needs to keep machine/SM full)
* Too big a P reaches the limit on shared memory size

**Privatalization:**

* within each tile, one row per thread
* saves thread, no communication among threads 

**Lessons learned:** balance between max parallelizm & more work/thread

###SpMV
**Method 1: thread per row**

* Launches less thread, more work/thread, no communication
* Load imbalance 

**Method 2: thread per element**

* Insensititve of density of each row
* Needs syncthreads()

**Method 3: hybrid method 1 & 2 (Divide the matrix into two parts)**

* Part 1 has nearly identical density in each row: thread per row
* Part 2 has varying density:thread per element

**Lessons learned:**

* Keep all thread busy (avoid load imbalance)
* Closer communication is cheaper (GPU communicating with shared rather than host)

###Bread-First Traversal in Graphs
**Method 1:** Iterate until all vertices assigned a depth (use a bool flag)

* Parallelize on edges, 
* For each edge:
  * If one vertex has depth d, but the other hasn't been visited: other vertex gets d+1

Issue: parallel, memory behavior ok, no thread divergence, but work complexity too high

**Method 2:** Store adjacency matrix with CSR representation, set init frontier to 0

* Allocate space for next frontiers
* Copy each active edge list to the array
* Remove those already visited by compact
* Set depth of fourier to new depth

**Lessons learned:**

* choose efficient algorithm
* Irregular expansion/contraction: scan

###List Ranking
N nodes in a linear change, each node knows id of next chain, put nodes in order

**Method 1:**

* Init chum of each node to self
* Set chum = chum.next

**Method 2:**

* Set chum = chum.chum, so that chum is 1, 2, 4, 8… hops away
* Make a table with each col a node, each row the node 2k steps away
* Construct output from the table
* Scatter ops to get output

**Lessons learned:**

* Trade more work for fewer steps
* Power of scan

###Hashing
**Hash table**

* bucket: chain of values
* Chaining in parallel is bad for local imbalance/contention in construction process (two threads may both want to add to the same hashtable, and requires synchronization)

**Cuckoo hashing:**

* Multiple hash tables, only 1 item in each bucket
* 1st iter: all items using h1 and write into t1
* 2nd iter: those that fail use h2, t2
* If fail on last hashtable, go back to hashtable 1 and kickout the item already in h1, try to put the kicket out value into other tables

**Lessons learned:** parallel friendly data structure
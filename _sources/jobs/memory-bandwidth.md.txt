# Efficient use of memory bandwith on Betzy

## Memory - NUMA and ccNUMA

Betzy compute node is a 2-socket system running AMD EPYC 7742 64-Core
processors.  Each compute node on Betzy has 256 GiB of memory, organised in 8
banks of 32 GiB each. Every processor has four memory controllers, each
responsible for one bank. Furthermore, every virtual core in a processor is
assigned to one memory controller which results in different paths to access
memory. Memory accesses may have to traverse an intra-processor network
(another controller within the same processor is responsible for the memory
address being accessed) or an intra-node network (another controller of the
other processor is responsible for the memory address being accessed). This
memory organisation is referred to as non uniform memory access (NUMA) memory.

This means that although all CPU cores can access all RAM, **the speed of the
access will differ: some memory pages are closer to each CPU, and some are
further away**. In contrast to a similar Intel-based system, where each socket
is one NUMA node, Betzy has 4 NUMA nodes per socket, and 8 in total.

A NUMA node comprises of a memory bank and a subset of the virtual cores. The
best performance is achieved when processes and the memory they access
(frequently) are placed close to each other, or in other words, within one NUMA
node.

Additionally, the compute nodes implement cache coherent NUMA (ccNUMA) memory
which ensures that programs see a consistent memory image. Cache coherence
requires intra-node communication if different caches store the same memory
location.  Hence, the best performance is achieved when the same memory
location is not stored in different caches. Typically this happens when
processes need access to some shared data, e.g. at boundaries of regions they
iterate over. Limiting or even avoiding these accesses is often a challenge.


## Detailed information about NUMA nodes

More detailed information about the NUMA
architecture can be obtained as follows:
```
$ numactl -H

available: 8 nodes (0-7)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
node 0 size: 32637 MB
node 0 free: 30739 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
node 1 size: 32767 MB
node 1 free: 31834 MB
node 2 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
node 2 size: 32767 MB
node 2 free: 31507 MB
node 3 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
node 3 size: 32755 MB
node 3 free: 31489 MB
node 4 cpus: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
node 4 size: 32767 MB
node 4 free: 31746 MB
node 5 cpus: 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
node 5 size: 32767 MB
node 5 free: 31819 MB
node 6 cpus: 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
node 6 size: 32767 MB
node 6 free: 31880 MB
node 7 cpus: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255
node 7 size: 32767 MB
node 7 free: 31805 MB
node distances:
node   0   1   2   3   4   5   6   7
  0:  10  12  12  12  32  32  32  32
  1:  12  10  12  12  32  32  32  32
  2:  12  12  10  12  32  32  32  32
  3:  12  12  12  10  32  32  32  32
  4:  32  32  32  32  10  12  12  12
  5:  32  32  32  32  12  10  12  12
  6:  32  32  32  32  12  12  10  12
  7:  32  32  32  32  12  12  12  10
```


## Binding thread/processes to cores

The above picture is further complicated by the fact that within the individual
NUMA nodes the memory access time is also not uniform. This can be verified by
running the [STREAM benchmark](https://www.cs.virginia.edu/stream/ref.html). As
reported above, each NUMA node has 16 physical cores (e.g. node 0, cores 0-15).

Consider the following 2 STREAM experiments:

1. start 8 threads, bind them to cores 0-8
2. start 8 threads, bind them to cores 0,2,4,6,8,10,12,14

In terms of the OMP_PLACES directive the above is equivalent to:

1. OMP_PLACES="{0:1}:8:1" OMP_NUM_THREADS=8 ./stream
2. OMP_PLACES="{0:1}:8:2" OMP_NUM_THREADS=8 ./stream

On a standard Intel-based system the above two experiments would perform
identically. This is not the case on Betzy: the first approach is slower than the
second one:

| Experiment | Function | Best Rate MB/s | Avg time | Min time | Max time |
| ---------- | -------- | -------------- | -------- | -------- | -------- |
| 1          | Copy     | 37629.4        | 0.212833 | 0.212600 | 0.213007 |
| 1          | Triad    | 35499.6        | 0.338472 | 0.338032 | 0.338771 |
| 2          | Copy     | 42128.7        | 0.190025 | 0.189894 | 0.190152 |
| 2          | Triad    | 41844.4        | 0.287000 | 0.286777 | 0.287137 |

This shows that the memory access time is not uniform within a single NUMA node.

Interestingly, the peak achievable memory bandwidth also depends on the number
of cores used, and is maximized for lower core counts. This is confirmed by the
following STREAM experiments running on one NUMA node:

1. start 8 threads, bind them to cores 0,2,4,6,8,10,12,14
2. start 16 threads, bind them to cores 0-15

In terms of the OMP_PLACES directive the above is equivalent to:

1. OMP_PLACES="{0:1}:8:2" OMP_NUM_THREADS=8 ./stream
2. OMP_PLACES="{0:1}:16:1" OMP_NUM_THREADS=16 ./stream

The results are:

| Experiment | Function | Best Rate MB/s | Avg time | Min time | Max time |
| ---------- | -------- | -------------- | -------- | -------- | -------- |
| 1          | Copy     | 42126.3        | 0.190034 | 0.189905 | 0.190177 |
| 1          | Triad    | 41860.1        | 0.287013 | 0.286669 | 0.287387 |
| 2          | Copy     | 39675.8        | 0.201817 | 0.201634 | 0.201950 |
| 2          | Triad    | 39181.7        | 0.306733 | 0.306265 | 0.307508 |

The above test demonstrates that memory bandwidth is maximized when using 8 out of 16 cores per NUMA node.

The following experiments test the entire system:

1. start 64 threads, bind them to cores 0,2,...126
2. start 128 threads, bind them to cores 0,1,..127

In terms of the OMP_PLACES directive the above is equivalent to:

1. OMP_PLACES="{0:1}:64:2" OMP_NUM_THREADS=64 ./stream
2. OMP_PLACES="{0:1}:128:1" OMP_NUM_THREADS=128 ./stream

The results are:

| Experiment | Function | Best Rate MB/s | Avg time | Min time | Max time |
| ---------- | -------- | -------------- | -------- | -------- | -------- |
| 1          | Copy     | 334265.8       | 0.047946 | 0.047866 | 0.048052 |
| 1          | Triad    | 329046.7       | 0.073018 | 0.072938 | 0.073143 |
| 2          | Copy     | 315216.0       | 0.050855 | 0.050759 | 0.050926 |
| 2          | Triad    | 309893.4       | 0.077549 | 0.077446 | 0.077789 |

Hence on Betzy **memory bandwidth hungry applications will likely benefit from only using half of the cores (64)**.

Note however that it is not enough to just use 64 cores instead of 128. **It is
important to bind the threads/ranks to cores correctly**, i.e., to run on every
second core. So a correct binding is either 0,2,4,...,126, or 1,3,5,...,127.
The above assures that the application runs on both NUMA-nodes in the most
efficient way. If instead you run the application on cores 0-63 only, then it
will be running at 50% performance as only one NUMA node will be used.


## Monitoring thread/process placement

To monitor the placement of threads or processes/ranks the `htop` utility is
useful, just log in to a node running your application and issue the `htop`
command. By default  `htop` numbers the cores from 1 through 256. This can be
confusing at times (it can be changed in htop by pressing F2 and navigate to
display options and tick off count from zero).

The 0-127 (counting starts from zero) are the first one of the two SMT on the
AMD processor. The number of cores from 128 to 255 are the second SMT thread
and share the same executional units as do the first 128 cores.

Using this view it’s easy to monitor how the ranks and threads are allocated
and mapped on the compressor cores.


## Memory bandwidth sensitive MPI applications

Some MPI applications are very sensitive to memory bandwidth
and consequently will benefit from having fewer ranks per node than the number
of cores, like running only 64 ranks per node.

Using `#SBATCH --ntasks-per-node=64` and then launch using something like:
```
mpirun --map-by slot:PE=2 --bind-to core ./a.out
mpirun --map-by ppr:32:socket:pe=2 --bind-to core ./a.out
```

Tests have shown that more than 2x in performance is possible, but using twice
as many nodes. Twice the number of nodes will yield twice the aggregated
memory bandwidth. Needless to say also twice as many core hours.

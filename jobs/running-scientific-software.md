(running-scientific-software)=

# Running scientific software

## Introduction

Running scientific software on current modern HPC systems is a demanding task
even for experts. The system design is quite complex with interconnects, nodes,
shared and distributed memory on a range of levels, a large number of both
cores with threading within a core and not to mention all the software
libraries and run time settings.

This document provides some background information to better understand details
of various system compenents, and tries to illustrate how to make use of that
information for using modern HPC systems in the best possible way.


### Hardware

#### Interconnect - InfiniBand

A cluster (using Betzy as the example) contains a rather large number of nodes
(Betzy 1344) with an interconnect that enables efficient delivery of messages
(Message Passing Interface, MPI) between the nodes. On Betzy Mellanox
InfiniBand is used, in a HDR-100 configuration.  The HDR (High Data Rate)
standard is 200 Gbits/s and HDR-100 is half of this. This is a trade-off, as
each switch port can accommodate two compute nodes. All the compute nodes are
connected in a Dragonfly topology. (While not fully symmetrical, tests have
shown that the slowdown by spreading the ranks randomly around the compute
nodes had less than the tender specified 10%. Acceptance tests showed from 8 to
zero percent slow down depending on the application. Hence for all practical
purposes there is no need to pay special attention to schedule jobs within a
single rack/cell etc).


#### Processors and cores

Each compute node contains two sockets with a 64 core AMD processor per socket.
Every processor has 64 cores each supporting 2-way _simultaneous multithreading_
(SMT, see https://en.wikipedia.org/wiki/Simultaneous_multithreading for more
information). To not confuse these _threading_ capabilities in hardware with
threads in software (e.g., pthreads or OpenMP), we use the term _virtual core_
from now on.

For applications it looks as if every compute node has 256 independent
_virtual cores_ numbered from 0 to 255. Due to SMT, always two of these seemingly
independent virtual cores form a pair and share the executing units of a core. If both of
these two virtual cores are used in parallel by an application, the
application's performance will be the same as if it used only one virtual core
(and the other one is left idle) or if two different applications would use one
virtual core each, each of the two applications would achieve only half of the
performance of a core. To achieve the maximum performance from each core, it is
therefore important to pay attention to the mapping of processes to cores, that is,
**any two processes (or software threads) of an application must not share the
two virtual cores** or, in other words, one of the two virtual cores in a pair
shall be kept idle.

The following command provides information about the numbering of virtual cores:
`cat /proc/cpuinfo | sed '/processor\|physical id\|core id/!d' | sed 'N;N;s/\n/ /g'`
The first 128 entries (processor 0-127) correspond to the first virtual core.
Accordingly, the second 128 entries (processor 128-255) correspond to the second
virtual core. So, if one limits the placement of processes to processor
numbers 0-127, no process will share executional units with any other process.

Both Intel MPI and OpenMPI provide means to achieve such placements. See
examples below.


#### Memory - NUMA and ccNUMA

Each compute node has 256 GiB of memory being organised in 8 banks of 32 GiB
each. Every processor has four memory controllers each being responsible for
one bank. Furthermore, every virtual core in a processor is assigned to one memory
controller which results in different paths to access memory. Memory accesses
may have to traverse an intra-processor network (another controller within the
same processor is responsible for the memory address being accessed) or an
intra-node network (another controller of the other processor is responsible for
the memory address being accessed). This memory organisation is referred to as
Non Uniform Memory Access (NUMA) memory. A NUMA node comprises of a memory bank
and a subset of the virtual cores. The best performance is achieved when
processes and the memory they access (frequently) are placed close to each other, or
in other words, within one NUMA node.

Additionally, the compute nodes implement cache coherent NUMA (ccNUMA) memory
which ensures that programs see a consistent memory image. Cache coherence requires
intra-node communication if different caches store the same memory location.
Hence, the best performance is achieved when the same memory location is not
stored in different caches. Typically this happens when processes need access to
some shared data, e.g., at boundaries of regions they iterate over. Limiting or even
avoiding these accesses is often a challenge.

To display information about the NUMA nodes use the command `numactl -H`. More details on this
later.


#### AVX2 vector units

Each of the processor cores have two vector units, these are 256 bits wide and
can hence operate on four 64-bit floating point numbers simultaneously. With
two such units and selecting fused multiply add (FMA) up to 16 double precision
operations can be performed per clock cycle. (no program contain only FMA
instruction so these numbers are inflated). This yields a marketing theoretical
performance of frequency times number of cores times 16, 2.26 GHz * 128 * 16 =
4608 Gflops/s for a single compute node (or 6.2 Pflops/s for the complete
Betzy). In any case the vector units are important for floating point
performance, see the note on environment flag for MKL later.


## Slurm

### Introduction

The resource manager used by the Sigma2 systems is Slurm.

Requesting the correct set of resources in the Slurm job script is vital for
optimum performance. The job may be pure MPI or a hybrid using both MPI and
OpenMPI.

In addition we have seen that in some cases memory bandwidth is the performance
limiting factor and only 64 processes per compute node is requested.

All of this is set in the Slurm job script. The job script can be written in
any language that uses `#` as the comment sign. Bash is most common, but some
applications like NorESM use Python. Perl, Julia and R are other options. Here
is a simple Python example:
```python
#!/usr/bin/env python

#SBATCH --job-name=slurm
#SBATCH --account=nn9999k
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:0:5

import os

os.system("srun hostname")
```

Using Python, Perl, Julia or R opens up for far more programming within the run
script than what’s possible using bash.

Some examples for Betzy are given below, for details about Slurm see the
[documentation about Slurm](job_scripts/slurm_parameter.md).


### Storage for scratch during a run on Saga

```{note}
Scratch storage is only available on Saga.
See also [the overview of storage areas](/files_storage/clusters.md).
```

Scratch storage for read and write files or any short lived files or files that are read and written to during a run should reside on a scratch pad area, [storage on clusters](/files_storage/clusters.md).

On Saga there are two scratch areas for such files (Fram and Betzy only has shared scratch), one shared for all processes and nodes which is residing on the parallel file system and another which is local on each compute node. The latter is smaller and only accessible to processes running on that node.

There are benefits for both types depending on usage. The parallel file system is by nature slower for random read & write operations and metadata operations (handling of large number fo files), the local file system is far better suited for this. In addition the shared file system need to serve all users and placing very high metadata load on it make the file system slow for all users. However, not only does aggregate performance scale with the number of nodes used but it does not affect the other users when using local scratch.

Which kind of scratch file system to use is a trade-off, if you need sharing or a large amount of data (more than 250-300 GB) there is only one option, shared scratch. If on the other hand you have a lot of random IO or a large number of files then local scratch is much better suited.

All SLURM jobs get allocated a shared scratch file system pointed to by the variable $SCRATCH , but you need to ask for local scratch, like this example where I have asked for 100 Gigabytes of local scratch:
```
#SBATCH --gres=localscratch:100G
```
By including this in the job script each allocated node will have a local scratch area pointed to by the variable $LOCALSCRATCH, (on Saga the maximum you can ask for is about 300 GB). Please do not ask for more than what you actually need, other users might share the local scratch space with you.
 
Saga has spinning disks with limited performance (but more nodes give more performance) for local scratch, but newer systems will have memory chips based storage for localscratch and hence it's good practice to start using local scratch now.

### Pure MPI

```
#SBATCH --ntasks=4096
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
```

This will request 32 nodes with 128 ranks per compute nodes giving a total of 4
ki ranks. The `--ntasks` is not strictly needed (if missing, it will be
calculated from `--nodes` and `--ntasks-per-node`.)

To get the total number of cores in a pure MPI job script the environment
variable `$SLURM_NTASKS` is available.


### Hybrid MPI + OpenMP

```
#SBATCH --nodes=1200
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
```

This will request 1200 nodes placing 8 MPI ranks per node and provide 16 OpenMP
threads to each MPI rank, a total of 128 cores per compute node. 1200 times 128
is 153600 or 150 ki cores.

To get the total number of cores in a Hybrid MPI + OpenMP job script one can
multiply the environment variables `$SLURM_NTASKS` and `$SLURM_CPUS_PER_TASK`.

To generate a list of all the Slurm variables just issue an `env` command in
the job script  and all environment variables will be listed.


#### srun vs mpirun

Most if the times the `mpirun` command can be used. The `mpirun` sets up the
MPI environment and makes sure that everything is ready for the MPI function
`MPI_init()` when it’s called in the start of any MPI program.

As Slurm is built with MPI support srun will also set up the MPI environment.

Both mpirun and srun launch the executable on the requested nodes. While there
is a large range of opinions on this matter it’s hard to make a final statement
about which one is best. If you do development on small systems like your
laptop or stand alone system there is generally no Slurm and mpirun is the only
option, so mpirun will work on everything from Raspberry Pis through laptops to
Betzy.

Performance testing does not show any significant performance difference when 
launching jobs in a normal way. There are, however a lot of possible 
options to mpirun both OpenMPI and Intel MPI. Both environment flags and 
command line options. 



## Library settings

The Intel MLK library performs run time checks at startup to select the
appropriate *INTEL* processor. When it cannot work ut the processor type a
least common instruction set is set is selected yielding lower performance.  To
instruct MKL to use a more suitable instruction set a debug variable can be
set, e.g. `export  MKL_DEBUG_CPU_TYPE=5`.

```{note}
The MKL_DEBUG_CPU_TYPE flag does not work for Intel compiler distribution    
2020 and and newer.                                                          
```

Users are adviced to check if there is any performance difference between 
Intel 2019b and the 2020 versions. It's not adviced to mix the Compiler and
MLK versions, e.g. building using 2020 and then link with MKL 2019. 


#### Forcing MKL to use best performing routines
MKL issue a run time test to check for genuine Intel processor. If
this test fail it will select a generic x86-64 set of routines
yielding inferior performance. This is well documentated in
[Wikipedia](https://en.wikipedia.org/wiki/Math_Kernel_Library) and
remedies in [Intel MKL on AMD Zen](https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html).

Research have discovered that MKL call a function called
*mkl_serv_intel_cpu_true()* to check the current CPU. If a genuine
Intel processor is found it simply return 1. The solution is simply to
override this function by writing a dummy functions which always
return *1* and place this first in the search path.  The function is
simply: 
```c
int mkl_serv_intel_cpu_true() { 
	return 1; 
}
```
Save this into a file called *fakeintel.c* and compile it into a shared library 
using the following command: 
`gcc -shared -fPIC -o libfakeintel.so fakeintel.c`

To put the new shared library first in the search path we can use a preload environment variable:
`export LD_PRELOAD=<path to lib>/libfakeintel.so`
A suggestion is to place the new shared library in `$HOME/lib64` and using 
`export LD_PRELOAD=$HOME/lib64/libfakeintel.so` to insert the fake test function.

In addition the envionment variable *MKL_ENABLE_INSTRUCTIONS* can also have a significant effect. 
Setting the variable to AVX2 is adviced. Just changing it to AVX have a significant negative impact.
Setting it to AVX512 and launching it on AMD it does not fail, MKL probably do test if feature requesed 
is present. 

The following table show the recorded performance obtained with the HPL (the top500) test using a small problem size and 
a single Betzy node:

| Settings                                                   |   Performance     | 
|-----------------------------------------------------------:|:-----------------:|
| None                                                       |  1.2858 Tflops/s  |
| LD_PRELOAD=./libfakeintel.so                              |  2.7865 Tflops/s  |
| LD_PRELOAD=./libfakeintel.so MKL_ENABLE_INSTRUCTIONS=AVX  |  2.0902 Tflops/s  |
| LD_PRELOAD=./libfakeintel.so MKL_ENABLE_INSTRUCTIONS=AVX2 |  2.7946 Tflops/s  |


The reccomendation is to copy `libfakeintel.so to` to `$HOME/lib64` and to set the preload evironment variable accordingly :
`export LD_PRELOAD=$HOME/lib64/libfakeintel.so` or `export LD_PRELOAD=<path to lib>/libfakeintel.so` and to set 
`export MKL_ENABLE_INSTRUCTIONS=AVX2` 




## Memory architecture

Betzy compute node is a 2-socket system running AMD EPYC 7742 64-Core
processors. Each node is a NUMA (*Non-Uniform Memory Access*) system, which
means that although all CPU cores can access all RAM, the speed of the access
will differ: some memory pages are closer to each CPU, and some are further
away. In contrast to a similar Intel-based system, where each socket is one
NUMA node, Betzy has 4 NUMA nodes per socket, and 8 in total. The NUMA
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

The above picture is further complicated by the fact that within the individual
NUMA nodes the memory access time is also not uniform. This can be verified by
running the STREAM benchmark. As reported above, each NUMA node has 16 physical
cores (e.g. node 0, cores 0-15). Consider the following 2 STREAM experiments:

1. start 8 threads, bind them to cores 0-8
2. start 8 threads, bind them to cores 0,2,4,6,8,10,12,14

In terms of the OMP_PLACES directive the above is equivalent to:

1. OMP_PLACES="{0:1}:8:1" OMP_NUM_THREADS=8 ./stream
2. OMP_PLACES="{0:1}:8:2" OMP_NUM_THREADS=8 ./stream

On a standard Intel-based system the above two experiments would perform
identical. This is not the case on Betzy: the first approach is slower than the
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
following STREAM experiments running on one NUMA node

1. start 8 threads, bind them to cores 0,2,4,6,8,10,12,14
2. start 16 threads, bind them to cores 0-15

In terms of the OMP_PLACES directive the above is equivalent to

1. OMP_PLACES="{0:1}:8:2" OMP_NUM_THREADS=8 ./stream
2. OMP_PLACES="{0:1}:16:1" OMP_NUM_THREADS=16 ./stream

The results are:

| Experiment | Function | Best Rate MB/s | Avg time | Min time | Max time |
| ---------- | -------- | -------------- | -------- | -------- | -------- |
| 1          | Copy     | 42126.3        | 0.190034 | 0.189905 | 0.190177 |
| 1          | Triad    | 41860.1        | 0.287013 | 0.286669 | 0.287387 |
| 2          | Copy     | 39675.8        | 0.201817 | 0.201634 | 0.201950 |
| 2          | Triad    | 39181.7        | 0.306733 | 0.306265 | 0.307508 |

which shows that memory bandwidth is maximized when using 8 out of 16 cores per NUMA node.

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

Hence on Betzy memory bandwidth hungry applications will likely benefit from only using half of the cores (64).

Note however that it is not enough to just use 64 cores instead of 128. It is
crucial to bind the threads/ranks to cores correctly, i.e., to run on every
second core. So a correct binding is either 0,2,4,...,126, or 1,3,5,...,127.
The above assures that the application runs on both NUMA-nodes in the most
efficient way. If instead you run the application on cores 0-63 only, then it
will be running at 50% performance as only one NUMA node will be used.


## Running MPI applications


### Introduction

The MPI to use is selected at application build time. Two MPI implementations
are supported:
* Intel MPI
* OpenMPI

Behavior of Intel MPI can be adjusted through environment variables environment
variables, which start with *I_MPI*
(https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference.html)

OpenMPI uses both environment variables (which must be used when running
through *srun*) and command line options (for use with *mpirun*). Command line
options override both the config files and environment variables. For a
complete list of parameters run `ompi_info --param all all --level 9`, or see
the documentation (https://www.open-mpi.org/faq/?category=tuning).


### Binding

Since not all memory is equal some sort of binding to keep the process located on cores close to the memory is normally beneficial.


#### Intel MPI

Enable binding can be requested with an  environment flag, I_MPI_PIN=1 .

To limit the ranks to only the fist thread on SMT e.g. using only cores 0 to 127 set the Intel MPI environment variable I_MPI_PIN_PROCESSOR_EXCLUDE_LIST to 128-255, e.g.
`export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255`.


#### OpenMPI

The simplest solution is just to request binding at the command line:
`mpirun --bind-to core ./a.out`
In this case binding to core is requested.  To learn more about the binding options try issuing the following command : `mpirun --help binding` .


### Collective MPI operations optimisation

For OpenMPI the flag *OMPI_MCA_coll_hcoll_enable* to 0 to disable or 1 to
enable can have significant effect on the performance of your MPI application.
Most of the times it’s beneficial to enable it by including `export
OMPI_MCA_coll_hcoll_enable=1` in the run script.


### Running MPI only applications


For well behaved MPI applications the job scripts are relatively simple. The
only important thing to notice is that processes (MPI ranks)  should be mapped
with one rank per two SMT threads (also often referred to a physical cores).

For both Intel and Open -MPI the simplest command line would be `mpirun
./a.out` This will launch the MPI application with the default settings, while
not with optimal performance it will run the application using the resources
requested in the run script.


#### Intel MPI

The variable I_MPI_PIN_PROCESSOR_EXCLUDE_LIST as mentioned earlier is good to
keep the ranks only on one of the two threads per processor core.  Intel uses
an environment variable to achieve the binding : I_MPI_PIN=1


#### OpenMPI

For more optimal performance process binding can be introduced, like for
OpenMPI: `--bind-to core`


#### Memory bandwidth sensitive applications

Some applications are very sensitive to memory bandwidth
and consequently will benefit from having fewer ranks per node than the number 
of cores, like running only 64 ranks per node. 
Using `#SBATCH --ntasks-per-node=64` and then lauch using something like:
```
mpirun --map-by slot:PE=2 --bind-to core ./a.out
mpirun --map-by ppr:32:socket:pe=2 --bind-to core ./a.out
```
Tests have shown that more than 2x in performance is possible, but using twice 
as many nodes. Twice the number of nodes will yield twice the aggregated 
memory bandwith. Needless to say also twice as many core hours.  


### Running  MPI-OpenMP - Hybrid applications

For large core count a pure MPI solution is often not optimal. Like HPL (the
top500 test) the hybrid model is the highest performing case.

Most OpenMP or threaded programs respond to the environment variable
OMP_NUM_THREADS.  You want to set it or another variable like NPUS etc  set it
from a Slurm variable : `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` or
`export NCPUS=$SLURM_CPUS_PER_TASK`.

The mapping of ranks and OpenMP threads onto the cores on the compute node can
often be tricky. There are many ways of dealing with this, from the simplest
solution by just relying on the defaults to explicit placement of the ranks and
threads on precisely specified cores.


#### Intel MPI

There are a lot of environment variables to be used with Intel MPI, they all start with I_MPI
* I_MPI_PIN
* I_MPI_PIN_DOMAIN
* I_MPI_PIN_PROCESSOR_EXCLUDE_LIST
The variable I_MPI_PIN_DOMAIN is good when running hybrid codes, setting it to
the number of threads per rank will help the launcher to place the ranks
correct. Setting I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255 will make sure only
cores 0-127 are used for MPI ranks. This ensures that no two ranks
share the same physical core. 


#### OpenMPI

There is currently some issues with mapping of threads started by MPI
processes. These threads are scheduled/placed on the same core as the MPI rank
itself. The issue seems to be an openMP issue with GNU OpenMP. We are working to
resolve this issue. 


### Create a hostfile or machinefile

To make a host or machinefile a simple  srun command can be used:
`srun /bin/hostname | uniq`

A more complicated example is a nodelist file for the molecular mechanics application NAMD :
`srun /bin/hostname | uniq | awk -F\. 'BEGIN {print “group main”};{print "host ", $1}' > nodelist`


### Transport options OpenMPI

Most of the following is hidden behind some command line options, but in case
more information is needed about the subject of transport a few links will
provide more insight.

For detailed list of settings a good starting point is here :
https://www.open-mpi.org/faq/

OpenMPI 4.x uses UCX for transport, this is a communication library:
https://github.com/openucx/ucx/wiki/FAQ

Transport layer PML (point-to-point message layer) :
https://rivis.github.io/doc/openmpi/v2.1/pml-ob1-protocols.en.xhtml

Transport layer UCX :
https://www.openucx.org/ and https://github.com/openucx/ucx/wiki/UCX-environment-parameters

Collective optimisation library hcoll from Mellanox is also an option:
https://docs.mellanox.com/pages/viewpage.action?pageId=12006295

Setting the different devices and transports can be done using environment variables:
```
export UCX_IB_DEVICE_SPECS=0x119f:4115:ConnectX4:5,0x119f:4123:ConnectX6:5
export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=self,sm,dc
```

From the UCX documentation the list of internode transports include :
* rc
* ud
* dc

The last one is Mellanox scalable offloaded dynamic connection transport.  The
self is a loopback transport to communicate within the same process, while sm
is all shared memory transports. There are two shared memory transports
installed
* cma
* knem

Selecting cma or knem might improve performance for applications that uses a
high number of MPI ranks per node. With 128 cores and possibly 128 MPI ranks
per node the intra node communication is quite important.

Depending on your application’s communication pattern, point-to-point or
collectives the usage of Mellanox optimised offload collectives can have an
impact.


### Monitoring process/thread placement

To monitor the placement or ranks the `htop` utility is useful, just log in to
a node running your application and issue the `htop` command. By default  `htop`
numbers the cores from 1 through 256. This can be confusing at times (it can be
changed in htop by pressing F2 and navigate to display options and tick off
count from zero).

The 0-127 (assume you’ve ticked of the start to count from zero) are the first
one of the two SMT on the  AMD processor. The number of cores from 128 to 255
are the second SMT thread and share the same executional units as do the first
128 cores.

Using this view it’s easy to monitor how the ranks and threads are allocated
and mapped on the compressor cores.

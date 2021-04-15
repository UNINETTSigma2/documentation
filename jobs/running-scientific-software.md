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


## Slurm

### Introduction

Requesting the correct set of resources in the Slurm job script is vital for
optimum performance. The job may be pure MPI or a hybrid using both MPI and
OpenMPI.

In addition we have seen that in some cases memory bandwidth is the performance
limiting factor and only 64 processes per compute node is requested.

Some examples for Betzy are given below, for details about Slurm see the
[documentation about Slurm](job_scripts/slurm_parameter.md).


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

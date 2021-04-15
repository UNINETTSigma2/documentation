# Efficient use of processors and network on Betzy


## Interconnect - InfiniBand

A cluster (using Betzy as the example) contains a rather large number of nodes
(Betzy consists of 1344 nodes) with an interconnect that enables efficient
delivery of messages (message passing interface, MPI) between the nodes. On
Betzy, Mellanox InfiniBand is used, in a HDR-100 configuration. The HDR (high
data rate) standard is 200 Gbits/s and HDR-100 is half of this. This is a
trade-off, as each switch port can accommodate two compute nodes. All the
compute nodes are connected in a Dragonfly topology.

While not fully symmetrical, tests have shown that the slowdown by spreading
the ranks randomly around the compute nodes had less than the 10% specified by
the tender. Acceptance tests showed from 8 to zero percent slow-down depending
on the application. Hence **for all practical purposes there is no need to pay
special attention to schedule jobs within a single rack/cell etc**.


## Processors and cores

Each compute node on Betzy contains two sockets with a 64 core AMD processor per socket.
Every processor has 64 cores each supporting 2-way [simultaneous multithreading](https://en.wikipedia.org/wiki/Simultaneous_multithreading)
(SMT).
To not confuse these *threading* capabilities in hardware with
threads in software (e.g., pthreads or OpenMP), we use the term *virtual core*
from now on.

For applications it looks as if every compute node has 256 independent
*virtual cores* numbered from 0 to 255. Due to SMT, always two of these seemingly
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
```
cat /proc/cpuinfo | sed '/processor\|physical id\|core id/!d' | sed 'N;N;s/\n/ /g'
```

The first 128 entries (processor 0-127) correspond to the first virtual core.
Accordingly, the second 128 entries (processor 128-255) correspond to the second
virtual core. So, if one limits the placement of processes to processor
numbers 0-127, no process will share executional units with any other process.

Both Intel MPI and OpenMPI provide means to achieve such placements and below
we will show how.


## Available MPI implementations

The MPI to use is selected at application build time. Two MPI implementations
are supported:
* Intel MPI
* OpenMPI

Behavior of Intel MPI can be adjusted through environment variables environment
variables, which start with *I_MPI*
([more information](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference.html)).

OpenMPI uses both environment variables (which must be used when running
through *srun*) and command line options (for use with *mpirun*). Command line
options override both the config files and environment variables. For a
complete list of parameters run `ompi_info --param all all --level 9`, or see
[the documentation](https://www.open-mpi.org/faq/?category=tuning).


## Slurm: Requesting pure MPI or hybrid MPI + OpenMP jobs


### Pure MPI

```
#SBATCH --ntasks=4096
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
```

This will request 32 nodes with 128 ranks per compute nodes giving a total of 4096
ranks/tasks. The `--ntasks` is not strictly needed (if missing, it will be
calculated from `--nodes` and `--ntasks-per-node`.)

To get the total number of cores in a pure MPI job script the environment
variable `$SLURM_NTASKS` is available.

For well behaved MPI applications the job scripts are relatively simple. The
only important thing to notice is that processes (MPI ranks)  should be mapped
with one rank per two SMT threads (also often referred to a physical cores).

For both Intel- and OpenMPI the simplest command line would be `mpirun
./a.out` This will launch the MPI application with the default settings, while
not with optimal performance it will run the application using the resources
requested in the run script.


### Hybrid MPI + OpenMP

For large core count a pure MPI solution is often not optimal. Like HPL (the
top500 test) the hybrid model is the highest performing case.

```
#SBATCH --nodes=1200
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
```

This will request 1200 nodes placing 8 MPI ranks per node and provide 16 OpenMP
threads to each MPI rank, a total of 128 cores per compute node. 1200 times 128
is 153600 cores.

To get the total number of cores in a Hybrid MPI + OpenMP job script one can
multiply the environment variables `$SLURM_NTASKS` and `$SLURM_CPUS_PER_TASK`.

To generate a list of all the Slurm variables just issue an `env` command in
the job script and all environment variables will be listed.

Most OpenMP or threaded programs respond to the environment variable
`OMP_NUM_THREADS` and you can set it to the number of CPUs per task
set by Slurm: `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`

The mapping of ranks and OpenMP threads onto the cores on the compute node can
often be tricky. There are many ways of dealing with this, from the simplest
solution by just relying on the defaults to explicit placement of the ranks and
threads on precisely specified cores.


#### Intel MPI

There are a number of environment variables to be used with Intel MPI, they all start with I_MPI:
* `I_MPI_PIN`
* `I_MPI_PIN_DOMAIN`
* `I_MPI_PIN_PROCESSOR_EXCLUDE_LIST`
The variable `I_MPI_PIN_DOMAIN` is good when running hybrid codes, setting it to
the number of threads per rank will help the launcher to place the ranks
correctly. Setting `I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255` will make sure that only
cores 0-127 are used for MPI ranks. This ensures that no two ranks
share the same physical core. 


#### OpenMPI

There are currently some issues with mapping of threads started by MPI
processes. These threads are scheduled/placed on the same core as the MPI rank
itself. The issue seems to be an openMP issue with GNU OpenMP. We are working to
resolve this issue. 


## Binding/pinning processes

Since not all memory is equally "distant", some sort of binding to keep the
process located on cores close to the memory is normally beneficial.


### Intel MPI

Binding/pinning to cores can be requested with an environment flag, `I_MPI_PIN=1`.

To limit the ranks to only the fist thread on SMT e.g. using only cores 0 to 127 set the Intel MPI environment variable `I_MPI_PIN_PROCESSOR_EXCLUDE_LIST` to 128-255, e.g.:
```
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=128-255
```


### OpenMPI

The simplest solution is just to request binding at the command line:
```
mpirun --bind-to core ./a.out
```

To learn more about the binding options try issuing the following command:
```
mpirun --help binding
```

## Optimizing collective MPI operations

For OpenMPI, setting the variable `OMPI_MCA_coll_hcoll_enable` to 0 to disable
or 1 to enable can have a significant effect on the performance of your MPI
application.  Most of the times it is beneficial to enable it by including
`export OMPI_MCA_coll_hcoll_enable=1` in the run script.



## srun vs mpirun on Betzy

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
launching jobs in a normal way.


## Creating a hostfile or machinefile

Some application ask for a list of hosts to distribute the tasks
across nodes and cores.

To make a *host-* or *machinefile*, you can use `srun`:
```
srun /bin/hostname | uniq
```

A more complicated example is a *nodelist* file for the molecular mechanics application NAMD :
```
srun /bin/hostname | uniq | awk -F\. 'BEGIN {print “group main”};{print "host ", $1}' > nodelist
```


## Transport options for OpenMPI

Most of the following is hidden behind some command line options, but in case
more information is needed about the subject of transport a few links will
provide more insight.

For detailed list of settings a good starting point is here:
<https://www.open-mpi.org/faq/>

OpenMPI 4.x uses UCX for transport, this is a communication library:
<https://github.com/openucx/ucx/wiki/FAQ>

Transport layer PML (point-to-point message layer):
<https://rivis.github.io/doc/openmpi/v2.1/pml-ob1-protocols.en.xhtml>

Transport layer UCX:
<https://www.openucx.org/ and https://github.com/openucx/ucx/wiki/UCX-environment-parameters>

Collective optimisation library hcoll from Mellanox is also an option:
<https://docs.mellanox.com/pages/viewpage.action?pageId=12006295>

Setting the different devices and transports can be done using environment variables:
```
export UCX_IB_DEVICE_SPECS=0x119f:4115:ConnectX4:5,0x119f:4123:ConnectX6:5
export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=self,sm,dc
```

From the UCX documentation the list of internode transports include:
* rc
* ud
* dc

The last one is Mellanox scalable offloaded dynamic connection transport. The
self is a loopback transport to communicate within the same process, while *sm*
is all shared memory transports. There are two shared memory transports
installed
* cma
* knem

Selecting *cma* or *knem* might improve performance for applications that use a
high number of MPI ranks per node. With 128 cores and possibly 128 MPI ranks
per node the intra node communication is quite important.

Depending on the communication pattern of your application, the use of point-to-point or
collectives, the usage of Mellanox optimised offload collectives can have an
impact.

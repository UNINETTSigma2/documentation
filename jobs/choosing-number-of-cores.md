(choosing-number-of-cores)=

# How to choose the number of cores

```{warning}
- Asking for too few cores can lead to underused nodes or longer run time
- Asking for too many cores can mean wasted CPU resources
- Asking for too much can mean a lot longer queuing
```

```{contents} Table of Contents
:depth: 3
```


## Why it matters

We request resources from the scheduler (queuing system).  But the scheduler
cannot tell how long the job will run and what resources it will really
consume.  Just the fact that I am asking the scheduler for 40 cores does not
mean that the code will actually run in parallel and use all of them.

Just because a website says that code X can run in parallel or "scales well"
does not mean that it will scale well for the particular feature/system/input
at hand.

Therefore **it is important to verify and calibrate this setting** for your use
case before computing very many similar jobs. Below we will show few
strategies.

Note that **you don't have to go through this for every single run**. This is
just to calibrate a job type. If many of your jobs have similar resource
demands, then the calibration will probably be meaningful for all of them.


## Using top

While the job is running, find out on which node(s) it runs using `squeue --me`,
then `ssh` into one of the listed compute nodes and run `top -u $USER`. See {ref}`squeue` for further reference on `squeue`

Some clusters also have `htop` available which produces similar output as `top`
but with colors and possibly clearer overview.


## Timing a series of runs

Here is an example C code (`example.c`) which we can compile and test a bit:
```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

double compute_something(int n, int m)
{
    double s = 0.0;
    for(int i = 0; i < n; i++)
    {
        double f = rand();
        for(int j = 0; j < m; j++)
        {
            f = sqrt(f);
        }
        s += f;
    }
    return s;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const int k = 10000;

    double my_values[k];
    double buffer_recv[k];

    for(int j = 0; j < 750; j++)
    {
        for(int i = 0; i < k; i++)
        {
            my_values[i] = compute_something(1000/size, 10);
        }
        for(int l = 0; l < 1000; l++)
        {
            MPI_Alltoall(&my_values, 10, MPI_DOUBLE, buffer_recv, 10, MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
```

It does not matter so much what the code does. Here we wish to **find out how
this code scales** and what an optimum number of cores for it might be on our
system.

We can build our example binary with this script (`compile.sh`):
```
#!/usr/bin/env bash

module purge
module load foss/2022b

mpicc example.c -O3 -o mybinary -lm
```

Now take the following example script
(tested on Saga, please adapt the line containing `--account=nn____k` to reflect your project number):
```{code-block}
---
emphasize-lines: 8-9
---
#!/usr/bin/env bash

#SBATCH --account=nn____k

#SBATCH --job-name='8-core'
#SBATCH --time=0-00:10:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --ntasks=8
#SBATCH -o 8.out

module purge
module load foss/2022b

time srun ./mybinary
```

Run a series of calculations on 1, 2, 4, 8, 16, 32, 64, and 128 cores.

You might get the following timings:

| Number of cores | Time spent in mybinary |
|-----------------|------------------------|
|   1             |      7m21s             |
|   2             |      2m21s             |
|   4             |      1m15s             |
|   8             |      0m41s             |
|  16             |      0m27s             |
|  32             |  (technical problem)   |
|  64             |      0m46s             |
| 128             |      2m07s             |

Please try this. What can we conclude? And how can we explain it?

Conclusions:
- For this particular example it does not make sense to go much beyond 16 cores
- Above 8 cores communication probably starts to dominate over computation


## Using seff

`seff JOBID` is a nice tool which we can use on **completed jobs**.

Here we can compare the output from `seff` when using 4 cores and when using
8 cores.

Run with 4 cores:
```{code-block}
---
emphasize-lines: 8-9
---
Job ID: 10404723
Cluster: saga
User/Group: someuser/someuser
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 4
CPU Utilized: 00:04:45
CPU Efficiency: 91.35% of 00:05:12 core-walltime
Job Wall-clock time: 00:01:18
Memory Utilized: 933.94 MB (estimated maximum)
Memory Efficiency: 22.80% of 4.00 GB (1.00 GB/core)
```

Run with 8 cores:
```{code-block}
---
emphasize-lines: 8-9
---
Job ID: 10404725
Cluster: saga
User/Group: someuser/someuser
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 8
CPU Utilized: 00:05:06
CPU Efficiency: 86.93% of 00:05:52 core-walltime
Job Wall-clock time: 00:00:44
Memory Utilized: 1.84 GB (estimated maximum)
Memory Efficiency: 23.01% of 8.00 GB (1.00 GB/core)
```


## Using jobstats

Try it with one of your jobs:
```console
$ jobstats -j 10404723

Job 10404723 consumed 0.1 billing hours from project nn****k.

Submitted 2024-02-04T13:52:44; waited 0.0 seconds in the queue after becoming eligible to run.

Requested wallclock time: 10.0 minutes
Elapsed wallclock time:   1.3 minutes

Task and CPU statistics:
ID                 CPUs  Tasks  CPU util                Start  Elapsed  Exit status
10404723              4            0.0 %  2024-02-04T13:52:44   78.0 s  0
10404723.batch        4      1     0.7 %  2024-02-04T13:52:44   78.0 s  0
10404723.mybinary     4      4    95.7 %  2024-02-04T13:52:48   74.0 s  0

Used CPU time:   4.8 CPU minutes
Unused CPU time: 26.7 CPU seconds

Memory statistics, in GiB:
ID                  Alloc   Usage
10404723              4.0
10404723.batch        4.0     0.0
10404723.mybinary     4.0     0.9
```


## If it does not scale, what can be possible reasons?

Here are typical problems:
- At some point more time is spent communicating than computing
- Memory-bound jobs saturate the memory bandwidth
- At some point the non-parallelized code section dominates the compute time ([Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law))


## What is MPI and OpenMP and how can I tell?

These two parallelization schemes are very common (but there exist other schemes):
- [Message passing interface](https://en.wikipedia.org/wiki/Message_Passing_Interface):
  typically each task allocates its own memory, tasks communicate via messages.
  It is no problem to go beyond one node.
- [OpenMP](https://www.openmp.org/):
  threads share memory and communicate through memory.
  We cannot go beyond one node.


### How to tell if the code is using one of the two?

- If you wrote the software: then you probably know
- If it is written by somebody else:
  - It can be difficult to tell
  - Consult manual for the software or contact support (theirs or ours)
  - **If you have access to the source code**, `grep -i mpi` and `grep -i "omp "` the source code
  - Example: <https://github.com/MRChemSoft/mrchem> (uses both MPI and OpenMP)


### Python/R/Matlab

- Small self-written scripts are often not parallelized
- Libraries that you include in your scripts can use parallelization (e.g.
  `mpi4py` or `multiprocessing`)


### Code may call a library which is shared-memory parallelized

- Examples: BLAS libraries, NumPy, SciPy

Here is an example which you can try (`example.py`) where we compute a couple
of matrix-matrix multiplications using [NumPy](https://numpy.org/):
```python
import numpy as np

n = 10000

# run it multiple times, just so that it runs longer and we have enough time to
# inspect it while it's running
for _ in range(5):
    matrix_a = np.random.rand(n, n)
    matrix_b = np.random.rand(n, n)

    matrix_c = np.matmul(matrix_a, matrix_b)

print("calculation completed")
```

We will try two different job scripts and below we highlight where they differ.

Job script A (adapt `--account=nn____k`; this is tested on Saga):
```{code-block}
---
emphasize-lines: 10-11
---
#!/usr/bin/env bash

#SBATCH --account=nn____k

#SBATCH --job-name='example'
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1500M

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

module load SciPy-bundle/2023.02-gfbf-2022b

python example.py

env | grep NUM_THREADS
```

Job script B:
```{code-block}
---
emphasize-lines: 10
---
#!/usr/bin/env bash

#SBATCH --account=nn____k

#SBATCH --job-name='example'
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=1500M

#SBATCH --nodes=1
#SBATCH --tasks-per-node=4

module load SciPy-bundle/2023.02-gfbf-2022b

python example.py

env | grep NUM_THREADS
```

Run both examples and check the timing.
It can also be
interesting to log into the compute node while the job is running and using
`top -u $USER`. Can you explain what is happening here?

This was the job script with `--cpus-per-task=4`:
```{code-block} console
---
emphasize-lines: 11
---
$ seff 10404753

Job ID: 10404753
Cluster: saga
User/Group: someuser/someuser
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 4
CPU Utilized: 00:03:56
CPU Efficiency: 75.64% of 00:05:12 core-walltime
Job Wall-clock time: 00:01:18
Memory Utilized: 3.03 GB
Memory Efficiency: 51.75% of 5.86 GB
```

And this was the job script with the two export lines active:
This was the job script with `--tasks-per-node=4`:
```{code-block} console
---
emphasize-lines: 11
---
$ seff 10404754

Job ID: 10404754
Cluster: saga
User/Group: someuser/someuser
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 4
CPU Utilized: 00:03:55
CPU Efficiency: 24.79% of 00:15:48 core-walltime
Job Wall-clock time: 00:03:57
Memory Utilized: 3.02 GB
Memory Efficiency: 51.56% of 5.86 GB
```

The explanation is that the former job script automatically sets `OMP_NUM_THREADS=4`,
whereas the latter sets `OMP_NUM_THREADS=1`.

The morale of this story is that for Python and R it can be useful to verify
whether the script really uses all cores you give the job script. If it is
expected to use them but only runs on 1 core, check whether the required
environment variables are correctly set. Sometimes you might need to set them
yourself.

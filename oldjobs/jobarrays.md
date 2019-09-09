# Job arrays: 


## Many sequential jobs in parallel

In this example we wish to run many similar sequential jobs in parallel using job arrays. We take Python as an example but this does not matter for the job arrays:


[include](files/test.py)


Save this to a file called “test.py” and try it out:

```
$ python test.py

start at 15:23:48
sleep for 10 seconds ...
stop at 15:23:58
```

Good. Now we would like to run this script 16 times at (more or less) the same
time. For this we use the following


[include](files/slurm-job-array.sh)


Submit the script with `sbatch` and after a while you should see 16 output files in your submit directory:

```
$ ls -l output*txt

-rw------- 1 user user 60 Oct 14 14:44 output_1.txt
-rw------- 1 user user 60 Oct 14 14:44 output_10.txt
-rw------- 1 user user 60 Oct 14 14:44 output_11.txt
-rw------- 1 user user 60 Oct 14 14:44 output_12.txt
-rw------- 1 user user 60 Oct 14 14:44 output_13.txt
-rw------- 1 user user 60 Oct 14 14:44 output_14.txt
-rw------- 1 user user 60 Oct 14 14:44 output_15.txt
-rw------- 1 user user 60 Oct 14 14:44 output_16.txt
-rw------- 1 user user 60 Oct 14 14:44 output_2.txt
-rw------- 1 user user 60 Oct 14 14:44 output_3.txt
-rw------- 1 user user 60 Oct 14 14:44 output_4.txt
-rw------- 1 user user 60 Oct 14 14:44 output_5.txt
-rw------- 1 user user 60 Oct 14 14:44 output_6.txt
-rw------- 1 user user 60 Oct 14 14:44 output_7.txt
-rw------- 1 user user 60 Oct 14 14:44 output_8.txt
-rw------- 1 user user 60 Oct 14 14:44 output_9.txt
```

Observe that they all started (approximately) at the same time:

```
$ grep start output*txt

output_1.txt:start at 14:43:58
output_10.txt:start at 14:44:00
output_11.txt:start at 14:43:59
output_12.txt:start at 14:43:59
output_13.txt:start at 14:44:00
output_14.txt:start at 14:43:59
output_15.txt:start at 14:43:59
output_16.txt:start at 14:43:59
output_2.txt:start at 14:44:00
output_3.txt:start at 14:43:59
output_4.txt:start at 14:43:59
output_5.txt:start at 14:43:58
output_6.txt:start at 14:43:59
output_7.txt:start at 14:43:58
output_8.txt:start at 14:44:00
output_9.txt:start at 14:43:59
```

## Packaging smaller parallel jobs into one large 

There are several ways to package smaller parallel jobs into one large
parallel job. The preferred way is to use job arrays as above.  Browse the web for many examples on how to do it. Here we want to present a more pedestrian alternative which can give a lot of flexibility.

In this example we imagine that we wish to run a `bigmem` job with 5 MPI jobs at the same time, each using 4 tasks, thus totalling to 20 tasks. Once they finish, we wish to do a post-processing step:

```
#!/bin/bash

#SBATCH --account=nnNNNNk  # Substitute with your project name
#SBATCH --job-name=example
#SBATCH --ntasks=20
#SBATCH --time=0-00:05:00
#SBATCH --partition=bigmem
#SBATCH --mem-per-cpu=5000M

# Load MPI module
module purge
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module list

# The set of parallel runs:
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &

wait

# here a post-processing step
# ...

exit 0
```

Similarly, here is an example to run a `normal` job with 8 MPI jobs at
the same time, each using 16 tasks, thus totalling 128 tasks:

```
#!/bin/bash

#SBATCH --account=nnNNNNk  # Substitute with your project name
#SBATCH --job-name=example
#SBATCH --nodes=4
#SBATCH --time=00:05:00

# Load MPI module
module purge
module load OpenMPI/3.1.1-GCC-7.3.0-2.30
module list

# Needed for jobs in normal or optimist partition:
export SLURM_MEM_PER_CPU=1920

# The set of parallel runs:
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &

wait

# here a post-processing step
# ...

exit 0
```


A couple of notes:

- The wait command is important - the run script will only continue once
  all commands started with `&` have completed.
- It is possible to use `mpirun` instead of `srun`, although `srun` is
  recommended for OpenMPI.
- The `export SLURM_MEM_PER_CPU=1920` prior to the `srun` lines is
  needed for jobs in the `normal` or `optimist` partitions, because it
  is not possible to specify this to `sbatch` for such jobs.
  Alternatively, you can add `--mem-per-cpu=1920` or to the `srun`
  command lines (this only works with `srun`).  (1920 gives up to 32
  tasks per node.  If each task needs more than 1920 MiB per cpu, the
  number must be increased (and the number of tasks per node will be
  reduced).
- This technique does **not** work with IntelMPI, at least not when using
  `mpirun`, which is currently the recommended way of running IntelMPI jobs.

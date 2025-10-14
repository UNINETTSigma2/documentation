# GROMACS


GROMACS is a versatile package to perform molecular dynamics, i.e. simulate the
Newtonian equations of motion for systems with hundreds to millions of
particles.

[More information can be found on GROMACS' homepage.](https://www.gromacs.org)

## Running GROMACS

| Module     | Version     |
| :------------- | :------------- |
| GROMACS |2018.1-foss-2018b <br> 2019-foss-2018b <br> 2019.4-foss-2019b <br> 2020-foss-2019b <br> 2020-fosscuda-2019b <br> 2021-foss-2020b <br>|

To see available versions when logged in, issue the following command:
```bash
module spider gromacs
```

To use GROMACS, type:

```bash
module load GROMACS/<version>
```

specifying one of the available versions in the table above.

### Sample GROMACS Job Script

```bash
#!/bin/bash
#SBATCH --account=nn<NNNN>k
#SBATCH --job-name=topol
#SBATCH --time=1-0:0:0
#SBATCH --nodes=10

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module --quiet purge
module load GROMACS/<version>
module list

case=$SLURM_JOB_NAME

mpirun gmx_mpi mdrun $case.tpr
```

## Accelerating using GPUs
GROMACS is capable of speeding up by using attached accelerators, like the
Nvidia P100 GPU cards in Saga, or the A100 Nvidia GPUs in Betzy. Very little
adaptation is required on the user's side, as GROMACS is able to detect the GPUs
when available. Simply load a version of GROMACS with `fosscuda` in the name,
like `GROMACS/2020-fosscuda-2019b`, and then request GPUs with
`--partition=accel` and `--gpus-per-task=1`.

```{note}
GROMACS can use multiple GPUs, but these must be attached to separate MPI
ranks. By using the `--gpus-per-task` flag we can request one GPU per MPI rank.
Keep in mind that both Saga and Betzy only have 4 GPUs per node which limits
the number of ranks per node.
```

```bash
#!/bin/bash
#SBATCH --account=nn<NNNN>k
#SBATCH --job-name=topol
#SBATCH --time=1-0:0:0
## Total number of MPI ranks, can be more than 4, but should be multiple of 2
#SBATCH --ntasks=1
## Setup number of tasks and CPU cores per task
#SBATCH --ntasks-per-node=4
#sbatch --cpus-per-task=2  # Minimum number of cores per MPI rank for GROMACS
## GPU setup
#SBATCH --partition=accel
#SBATCH --gpus-per-task=1

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

module --quiet purge
module load GROMACS/2020-fosscuda-2019b
module list

case=$SLURM_JOB_NAME

mpirun gmx_mpi mdrun $case.tpr
```

```{note}
In the above job script we combined `--ntasks`, `--ntasks-per-node`,
`--cpus-per-task` and `--gpus-per-task`. This might seem counter intuitive, but
we did it for good reason.

First, by combining `--ntasks` and `--ntasks-per-node` the `--ntasks` takes
precedence and determines the number of MPI ranks to start. The
`--ntasks-per-node` then acts as limitation, determining the maximum number of
tasks per node
([reference](https://slurm.schedmd.com/sbatch.html#OPT_ntasks-per-node)). This
means that if we asked for `--ntasks=6` with `--ntasks-per-node=4` we would
still get 6 MPI ranks, but Slurm would have to reserve two nodes for us.

We then used `--cpus-per-task`, this was done since GROMACS requires at least
two threads per MPI rank so that each MPI rank has one computation thread and
one communication thread. We could give GROMACS more CPUs per MPI rank, since
GROMACS supports shared memory parallelization in addition to GPU acceleration,
but that is something that each project/experiment needs to test for themselves
to determine the utility.

Lastly, all of this combined ensures that if we want to use multiple GPUs
(which GROMACS support) we can simply increase the number of `--ntasks` and all
other parameters will be correct. There will only ever be 4 tasks per node
which corresponds to the number of GPUs per node and each of these MPI ranks
will have the necessary cores to run all bookkeeping tasks.
```

Using accelerators can give a nice speed-up, depending on the problem. As an
example we modeled a box of water with constant temperature and pressure. To
measure the difference in performance we compared a full CPU node on Saga with
a single GPU.

| Configuration | Wall time (s) | ns/day | Speed-up |
|:--------------|:--------------|:-------|---------:|
| 40 CPU cores  | 1892          | 456.427| `1x`     |
| 1 GPU + 2 CPU cores | 823     | 1049.088| `2.3x`  |

## License Information

GROMACS is available under the [GNU Lesser General Public License
(LGPL)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html), version 2.1.

It is the user's responsibility to make sure they adhere to the license
agreements.

## Citation

When publishing results obtained with the software referred to, please do check
the developers web page in order to find the correct citation(s).

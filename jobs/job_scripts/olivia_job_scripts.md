---
orphan: true
---

(job-scripts-on-olivia)=

<!-- TODO: 1.0 fill out this page -->

# Olivia job scripts

Here you can find job script examples for the Olivia HPC system.

## MPI jobs (CPU)

This is an example script for a multi-node MPI job on Olivia's CPU nodes, demonstrating Cray-specific environment variables.

```bash
#!/bin/bash
#SBATCH --account=nnXXXXk     # Your project
#SBATCH --job-name=Mandelbrot_MPI # A name for your MPI job
#SBATCH --time=0-0:05:00      # Total maximum runtime (e.g., 5 minutes)
#SBATCH --ntasks=16           # Total number of MPI tasks
#SBATCH --nodes=4             # Number of nodes to use
#SBATCH --partition=normal    # Run on the normal CPU partition
#SBATCH --cpus-per-task=1     # 1 CPU core per MPI task (adjust if using hybrid MPI/OpenMP)
#SBATCH --mem-per-cpu=3G      # Amount of CPU memory per CPU core


# Load necessary modules for your compilers and MPI library (e.g., PrgEnv-gnu, cray-mpich)
# module load PrgEnv-gnu cray-mpich # Example modules

# Run your MPI application
srun ./my_mpi_program
```



## Accel jobs (GPU)

This is an example script for a GPU job on Olivia. You must specify the `accel` partition and request specific GPU resources.

```bash
#!/bin/bash
#SBATCH --account=nnXXXXk     # Your project
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=05:10:00  
#SBATCH --account=nnXXXXk
#SBATCH --job-name=my-job-name
#SBATCH --output=my-job-name-%j.out     
#SBATCH --error=my-job-name-%j.err

# Load necessary modules (e.g., CUDA toolkit and programming environment)
module load PrgEnv-nvidia # Example programming environment
module load cuda/12.2 # Example CUDA version

# Run your GPU application
srun ./my_gpu_program
```

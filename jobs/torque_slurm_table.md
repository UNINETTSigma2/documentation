Quick Guide to translate PBS/Torque to SLURM
============================================

  User commands              PBS/Torque              SLURM
  -------------------------- ----------------------- -------------------------------
  Job submission             qsub \[filename\]       sbatch \[filename\]
  Job deletion               qdel \[job\_id\]        scancel \[job\_id\]
  Job status (by job)        qstat \[job\_id\]       squeue --job \[job\_id\]
  Full job status (by job)   qstat -f \[job\_id\]    scontrol show job \[job\_id\]
  Job status (by user)       qstat -u \[username\]   squeue --user=\[username\]

  Environment variables   PBS/Torque          SLURM
  ----------------------- ------------------- ------------------------
  Job ID                  \$PBS\_JOBID        \$SLURM\_JOBID
  Submit Directory        \$PBS\_O\_WORKDIR   \$SLURM\_SUBMIT\_DIR
  Node List               \$PBS\_NODEFILE     \$SLURM\_JOB\_NODELIST

  Job specification       PBS/Torque                      SLURM
  ----------------------- ------------------------------- -----------------------------------------------
  Script directive        \#PBS                           \#SBATCH
  Job Name                -N \[name\]                     --job-name=\[name\] OR -J \[name\]
  Node Count              -l nodes=\[count\]              --nodes=\[min\[-max\]\] OR -N \[min\[-max\]\]
  CPU Count               -l ppn=\[count\]                --ntasks-per-node=\[count\]
  CPUs Per Task                                           --cpus-per-task=\[count\]
  Memory Size             -l mem=\[MB\]                   --mem=\[MB\] OR --\`mem-per-cpu=\[MB\]
  Wall Clock Limit        -l walltime=\[hh:mm:ss\]        --time=\[min\] OR --time=\[days-hh:mm:ss\]
  Node Properties         -l nodes=4:ppn=8:\[property\]   --constraint=\[list\]
  Standard Output File    -o \[file\_name\]               --output=\[file\_name\] OR -o \[file\_name\]
  Standard Error File     -e \[file\_name\]               --error=\[file\_name\] OR -e \[file\_name\]
  Combine stdout/stderr   -j oe (both to stdout)          (Default if you don't specify --error)
  Job Arrays              -t \[array\_spec\]              --array=\[array\_spec\] OR -a \[array\_spec\]
  Delay Job Start         -a \[time\]                     --begin=\[time\]



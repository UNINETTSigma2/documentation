# Common Job Failures

Though users run different types of jobes on the clusters , there are a common set of reasons
that many jobs fail to run successfully.Here we will explain  some of these common failure modes
 and  steps to fix them.

## Running Out of Memory

Often, jobs can fail due to insufficient amount of memory being requested. This failure might 
appear in a SLURM error.
This can be corrected in the sbatch script by increase the amount of memory.

## Disk Quota Exceeded

Since the clusters are shared resources, we have quotas in place to prevent any group from
 using too much disk space. For more details to see how to inspect your quota please
 see [here.](https://documentation.sigma2.no/files_storage/clusters.html) 
When a group or user reaches the quota, files cannot  be created due to the cause 'Disk Quota Exceeded'. 
This will often kill jobs that need to write output or log files.
 
There are different quota for $HOME and $PROJECT. Additionally there is a File count quota that 
restrict the number of files that can be created. If the $HOME is at maximum capacity , please 
move some files to the $PROJECT or delete unnecessary files. For further details please see [here.](https://documentation.sigma2.no/files_storage/clusters.html)


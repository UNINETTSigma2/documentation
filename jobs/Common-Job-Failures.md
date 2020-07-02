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
 
There could be different quota settings for $HOME, $PROJECT and any other area in addition to a File count quota. If some of the areas are at the maximum capacity, consider deleting or moving some of the files. Please find more details [here.](https://documentation.sigma2.no/files_storage/clusters.html#frequently-asked-questions)


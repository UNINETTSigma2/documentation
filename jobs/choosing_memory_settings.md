

# How to choose the right amount of memory

##  General Note

It is important to make sure that your jobs use the right amount of memory and the right number of CPUs in order to help you and others using the hpc machines utilize these resources more efficiently, and in turn get work done more speedily. Remember to check the [Slurm documentation](https://slurm.schedmd.com/squeue.html#lbAG), [job types](https://documentation.sigma2.no/jobs/choosing_job_types.html),[Queue system concepts](https://documentation.sigma2.no/jobs/submitting/queue_system_concepts.html) and [HPC machines](https://documentation.sigma2.no/hpc_machines/hardware_overview.html) to verify that you are submitting the right job to the right partition and right hardware.

We recommend the users to run a test job and  then find out how many resources is used by the test job in order to effectively fathom how much resources a job needs. Recommended to ask ~20% extra time and memory than that you expect your job to need.

Here are some examples of how to gauge your CPU and Memory usage so you can ensure that you use your resources effeiciently.

##Completed Jobs

Slurm records statistics for every job, including how much memory and CPU was used. You can find out the slurm-JobId.out in the Slurm Job's work directory.

## Running Jobs

See the details [here](https://documentation.sigma2.no/jobs/monitoring.html)



# Packaging smaller parallel jobs into one large job

There are several ways to package smaller jobs into one large parallel job. The preferred way is to use [Job Arrays](jobarrays.md). Here we want to present a more pedestrian alternative which can give a lot of flexibility.

In this example we imagine that we wish to run 8 MPI jobs at the same time, each using 16 tasks, thus totalling to 128 tasks. Once they finish, we wish to do a post-processing step and then resubmit another set of 8 jobs with 16 tasks each:

[include](files/slurm-smaller-jobs.sh)

The `wait` commands are important here - the run script will only continue once all commands started with `&` have completed.


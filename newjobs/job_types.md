# Job Types
Describe the different types of jobs on each cluster, what they are supposed
to be used for, and which limits they have.  Perhaps in a separate page for
each cluster.  Use a standardized format.

http://hpc.uit.no/en/latest/jobs/partitions.html

## Fra Abel

General Job Limitations

Default values when nothing is specified:

    1 core (CPU)

The rest (time, mem per cpu, etc.) must be specified.

Limits

    Each project has its own limit of the number of concurrent used CPUs. See qsumm --group. (Notur projects share the same limit.)
    The max wall time is 1 week, except for jobs in the hugemem or long partitions, where it is 4 weeks. (definition of wall time)
    Max 400 submitted jobs per user at any time.

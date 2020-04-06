# Scheduling

Jobs are scheduled based on their priority.  In addition, lower
priority jobs can be back-filled if there are idle resources.

Note that job priority is only affected by the [job
type](choosing_job_types.md) and how long the job has been pending in the
queue.  Notably, job size and fair share usage does _not_ affect the
priorities.

## Job Priorities

The priority setup has been designed to be as predictable and easy to
understand as possible, while trying to maximize the utilisation of the
cluster.

The principle is that a job's priority increases 1 point per minute the job is
waiting in the queue[^1], and once the priority reaches 20,000, the job gets a
reservation in the future.  It can start before that, and before it gets a
reservation, but only if it does not delay any jobs with reservations.

The different job types start with different priorities:

- _devel_ jobs start with 20,000, so get a reservation directly
- _short_ jobs start with 19,880, so get a reservation in 2 hours
- _normal_, _bigmem_, _accel_ and _preproc_ jobs start with 19,760, so
  get a reservation in 4 hours
- "Unpri" _normal_, _bigmem_, _accel_ and _preproc_ jobs[^2] start
  with 19,040, so get a reservation in 16 hrs
- _optimist_ jobs start with 1 and end at 10,080, so they never get a reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other lower priority job should be able delay it.  But
before it gets a reservation, the queue system backfiller is free to
start any other job that can start now, even if that will delay the
job.  Note that there are still factors that can change the estimated
start time, for instance running jobs that exit sooner than expected,
or nodes failing.


## Job Placement on Fram

Note that this section _only_ applies to Fram!

The compute nodes on Fram are divided into four groups, or *islands*.  The
network bandwidth within an island is higher than the throughput between
islands.  Some jobs need high network throughput between its nodes, and will
usually run faster if they run within a single island.  Therefore, the queue
system is configured to run each job within one island, if possible.  See
[Job Placement on Fram](fram_job_placement.md) for details and for how this can
be overridden.


## Footnotes

[^1]: Currently, only the priority of 10 jobs for each user within each project increase with time.  As jobs start, more priorities start to increase.  This is done in order to avoid problems if a user submits a large amount of jobs over a short time.  Note that the limit is per user and project, so if a user has jobs in several projects, 10 of the user's jobs from each project will increase in priority at the same time.  This limit might change in the future.

[^2]: In some cases when a project applies for extra CPU hours because it has spent its allocation, it will get "unprioritized" hours, meaning their jobs will have lower priority than standard jobs (but still higher than `optimist` jobs).

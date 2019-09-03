# FIXME


## Job Priorities

The priority setup has been designed to be as predictable and easy to
understand as possible, while trying to maximize the utilisation of the
cluster.

The principle is that a job's priority increases 1 point per minute the job is
waiting in the queue, and once the priority reaches 20,000, the job gets a
reservation in the future.  It can start before that, and before it gets a
reservation, but only if it does not delay any jobs with reservations.

The different job types start with different priorities:

- `devel` jobs start with 20,000, so get a reservation directly
- `normal`, `bigmem` and `preproc` jobs start with 19,760, so get a
  reservation in 4 hours
- "Unpri" `normal`, `bigmem` and `preproc` jobs(*) start with 19,040, so get a reservation in 16 hrs
- `optimist` jobs start with 1 and end at 10,080, so they never get a reservation

The idea is that once a job has been in the queue long enough to get a
reservation, no other lower priority job should be able delay it. But before
it gets a reservation, the queue system backfiller is free to start any other
job that can start now, even if that will delay the job.  Note that there are
still factors that can change the estimated start time, for instance running
jobs that exit sooner than expected, or nodes failing.

(*) In some cases when a project applies for extra CPU hours because it has
spent its allocation, it will get "unprioritized" hours, meaning their jobs
will have lower priority than standard jobs (but still higher than `optimist`
jobs).


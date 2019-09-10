# Job Placement on Fram

The compute nodes on Fram are divided into four groups, called
*islands*.  Each island has about the same number of nodes.  The
Infiniband network throughput ("speed") within an island is higher than
the throughput between islands.  Some jobs need high network throughput
between its nodes, and will usually run faster if they run within a
single island.

## Default Setup

Therefore, the queue system is configured to run each job within one
island, if that does not delay the job too much.  It works like this:
When a job is submitted, the queue system lets the job wait until there
are enough free resources so that it can run within one island.  If this
has not happened when the job has waited 7 days[^1], the job will be
started on more than one island.

## Overriding the Setup

The downside of requiring that all nodes belonging to a job should be in the
same island, is that the job might have to wait longer in the queue,
especially if the job needs many nodes.  Some jobs do not need high network
throughput between its nodes.  For such jobs, you can override the setup,
either for individual jobs or for all your jobs.

### Individual Jobs

For individual jobs, you can use the switch `--switches=N[@time]` *on the
command line* when submitting the job, where *N* is the maximal number of
islands to use (1, 2, 3 or 4), and *time* (optional) is the maximum time to
wait.  See `man sbatch` for details.  Two examples:

    --switches=2          # Allow two islands
    --switches=1@4-0:0:0  # Change max wait time to 4 days

The maximal possible wait time to specify is 28 days[^1].  *A longer time
will silently be truncated to 28 days!*

Note that putting this option in an `#SBATCH` line in the job script will
**not** work (it will silently be overridden by the environment variables we
set to get the default behaviour)!

On the other hand, you might want to guarantee that your job never,
ever, starts on more than one island.  The easiest way to do that is to
specify `--constraint=[island1|island2|island3|island4]` instead (this option
can be used either on the command line or in the job script).

### Changing the Defaults

For changing the default for your jobs, you can change the
followin environment variables:

- `SBATCH_REQ_SWITCH`: Max number of islands for `sbatch` jobs.
- `SALLOC_REQ_SWITCH`: Max number of islands for `salloc` jobs.
- `SRUN_REQ_SWITCH`: Max number of islands for `srun` jobs.
- `SBATCH_WAIT4SWITCH`: Max wait time for `sbatch` jobs.
- `SALLOC_WAIT4SWITCH`: Max wait time for `salloc` jobs.
- `SRUN_WAIT4SWITCH`: Max wait time for `srun` jobs.

(`salloc` and `srun` jobs are interactive jobs; see
[Interactive Jobs](interactive_jobs.md).)  As above, the maximal possible wait
time to specify is 28 days[^1], and any time longer than that will *silently be
truncated*.  The change takes effect for jobs submitted after you change the
variables.  For instance, to change the default to allow two islands, and wait
up to two weeks:

```bash
export SBATCH_REQ_SWITCH=2
export SALLOC_REQ_SWITCH=2
export SRUN_REQ_SWITCH=2
export SBATCH_WAIT4SWITCH=14-00:00:00
export SALLOC_WAIT4SWITCH=14-00:00:00
export SRUN_WAIT4SWITCH=14-00:00:00
```

Note that we do *not* recommend that you unset these variables.  If you want
your jobs to start on any nodes, whichever island they are on, simply set
`*_REQ_SWITCH` variables to 4.  Specifically, if you unset the
`*_WAIT4SWITCH` variables, they will default to 28 days[^1].  Also, in the
future we might change the underlying mechanism, in which case unsetting these
variables will have no effect (but setting them will).

## Footnotes

[^1]: The limits might change in the future.

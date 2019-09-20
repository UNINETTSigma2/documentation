# Interactive Jobs

Sometimes you might want to test or debug a calculation interactively,
but running interactively on the login node is discouraged and not an
option.

What you can do instead is to ask the queue system to allocate compute
resources for you and once assigned, you can run the job(s)
interactively for as long as requested.  The examples below are for
_devel_ jobs, but the procedure also holds for the other job types
except _optimist_ jobs (see [Job Types](job_types.md) for more info).

On **Fram**:

    srun --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount --pty bash -i

On **Saga**:

    srun --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount --pty bash -i

When you are done, simply exit the shell (`exit`, `logout` or `^D`) to
end the job.  (Note to users coming from Abel: there is no `qlogin`
command on Fram or Saga.)

**Note: Right now, due to a bug in the current version of Slurm, the
Saga example does not work.  Slurm will be upgraded shortly, fixing
this bug**

The arguments between `srun` and `--pty` could be any arguments you
would have given to `sbatch` when submitting a non-interactive
job. However, `--qos=devel` is probably a good idea to avoid waiting
too long in the queue.

**Note that interactive jobs dies when you log out from the login
machine**, so unless you have very long days in office (or elsewhere,
for that matter), specifying more than 6-8 hrs runtime is not very
useful.  An alternative is to start the job in a `tmux` session.

## GUI Commands

It is possible to run X commands, i.e., programs with a graphical user
interface (GUI), in interactive jobs.

First, you must make sure that you have turned on *X forwarding* when logging
in to the cluster.  With `ssh` from a Linux or MacOS machine, you do this with
the `-Y` flag, e.g.:

    ssh -Y fram.sigma2.no

or

    ssh -Y saga.sigma2.no

Check that the X forwarding works by running a graphical command like `emacs &`
and verify that it sets up a window.  (Note that due to network latency, it
can take a long time to set up a window.)

To be able to run X commands in interactive jobs, add the argument `--x11`
(note: lower case `x`) to `srun`, like this:

On **Fram**:

    srun --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount --x11 --pty bash -i

On **Saga**:

    srun --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount --x11 --pty bash -i

## Alternative Method: `salloc`

An alternative to using `srun ... --pty bash -i` is to use

    salloc --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount

(and similar on Saga).

As with `srun`, the arguments to `salloc` can be any you would have
given to `sbatch`.  `salloc` will also give you an interactive shell, but note
that the shell **will be running on the login node**.  That means that you
*must* start all calculations with `srun` or `mpirun` or equivalent, to make
sure they run on the allocated compute node(s).  For this reason, the `srun`
method is probably preferrable in most cases (it also works more like a batch
job).  However, there might be situations when the `srun` method doesn't work
properly, in which case you can try with `salloc`.

With this method, you can run X commands without adding any switch to the
`salloc` command, but note again that the commands will be run **on the login
node**.  In some cases, this is what you want (see for instance
[interactive TotalView debugging](../development/debugging.md#debugging-interactive)).
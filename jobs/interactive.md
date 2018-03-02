# Interactive Jobs

Sometimes you might want to test or debug a calculation interactively but
running interactively on the login node is discouraged and not an option.

What you can do instead is to ask the batch system to allocate a node for you
and once it is assigned to you, you can run interactively on the node for up to 1 hour:

    $ srun --nodes=1 --time=01:00:00 --account=nnXXXXk --qos=devel --pty bash -i

The arguments between `srun` and `--pty` could
be any arguments you would have given to `sbatch` when submitting a
non-interactive job. However, `--qos=devel` is probably a good idea to avoid
waiting too long in the queue.

## GUI Commands

It is possible to run X commands, i.e., programs with a graphical user
interface (GUI), in interactive jobs.

First, you must make sure that you have turned on *X forwarding* when logging
in to the cluster.  With `ssh` from a Linux or MacOS machine, you do this with
the `-Y` flag, e.g.:

    $ ssh -Y fram.sigma2.no

Check that the X forwarding works by running a graphical command like `emacs &`
and verify that it sets up a window.  (Note that due to network latency, it
can take a long time to set up a window.)

To be able to run X commands in interactive jobs, add the argument `--x11`
(note: lower case `x`) to `srun`, like this:

    $ srun --nodes=1 --time=01:00:00 --account=nnXXXXk --qos=devel --x11 --pty bash -i

## Alternative Method: `salloc`

An alternative to using `srun ... --pty bash -i` is to use

    $ salloc --nodes=1 --time=01:00:00 --account=nnXXXXk --qos=devel

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

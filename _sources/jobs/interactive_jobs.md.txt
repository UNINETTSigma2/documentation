(interactive-jobs)=

# Interactive jobs

Sometimes you might want to test or debug a calculation interactively,
but **running interactively on the login node is discouraged and not an
option**.


```{contents} Table of Contents
```


## Requesting an interactive job

Instead of running on a login node, you can ask the queue system to
allocate compute resources for you, and once assigned, you can run
commands interactively for as long as requested.  The examples below
are for _devel_ jobs, but the procedure also holds for the [other job
types ](choosing_job_types.md) except _optimist_ jobs.

On **Saga** and **Olivia**:
```
salloc --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount
```

On **Betzy**:
```
salloc --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount
```

This will allocate resources, and start a shell on a compute node.
When you are done, simply exit the shell (`exit`, `logout` or `^D`) to
end the job.

The arguments to `salloc` (or `srun`) could be any arguments you
would have given to `sbatch` when submitting a non-interactive
job. However, `--qos=devel` is probably a good idea to avoid waiting
too long in the queue.

**Note that interactive jobs stop when you log out from the login
node**, so unless you have very long days in office (or elsewhere, for
that matter), specifying more than 6-8 hours runtime is not very
useful. An alternative is to start the job in a `tmux` session (see
below).


### GPU nodes on Olivia
The GPU nodes on Olivia (part of the `accel` partition use `ARM64` CPU
architecture instead of on our systems more common `x86` one.
To compile software or build containers that should run on the `accel`
partition, you can use an interactive session.

To ensure compatibility, you can use an **interactive session** to compile or test your software directly on the target architecture.

To start an interactive session on one of the Nvidia Grace-Hopper 200 nodes, use the following command:

```bash
salloc --nodes=1 --time=00:30:00 --qos=devel --partition=accel --account=YourAccount --mem=110G --cpus-per-task=70 --gpus=1
```

- **`--nodes=1`**: Requests one node.
- **`--time=00:30:00`**: Allocates a maximum runtime of 30 minutes.
- **`--qos=devel`**: Uses the development quality of service (QOS) for short jobs.
- **`--partition=accel`**: Use GPU partition.
- **`--account=YourAccount`**: Replace `YourAccount` with your project or account name.
- **`--mem=110G`**: Allocates 110 GB of memory.
- **`--cpus-per-task=70`**: Allocates 70 CPU cores for the task.
- **`--gpus=1`**: Allocates one GPU.


## Graphical user interface in interactive jobs

It is possible to run X commands, i.e., programs with a graphical user
interface (GUI), in interactive jobs. This allows you to get graphical
output back from your job running on a login node.  (Note that
currently, this has not been activated on Betzy.)

First, you must make sure that you have turned on *X forwarding* when logging
in to the cluster.  With `ssh` from a Linux or MacOS machine, you do this with
the `-Y` flag, e.g.:
```
$ ssh -Y saga.sigma2.no
```

Check that the X forwarding works by running a graphical command like `xeyes`
and verify that it sets up a window.  (Note that due to network latency, it
can take a long time to set up a window.)

To be able to run X commands in interactive jobs, add the argument `--x11`
(note the lowercase `x`) to `salloc`, like this:

On **Saga** and **Olivia**:
```
$ salloc --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount --x11
```


## Running the shell or a command on the login node

For some applications (see for instance {ref}`totalview_debugging`),
it is preferrable to have the shell or a command running on the login
node instead of on the compute node(s).

This can be achieved by just adding `bash` or the command to the end of
the `salloc` command line, i.e.,
```
$ salloc <options> bash
```
or
```
$ salloc <options> <command>
```

Note that the shell **will be running on the login node**.  That means
that you *must* start all calculations with `srun` or `mpirun` or
equivalent, to make sure they run on the allocated compute node(s).


## Keeping interactive jobs alive

Interactive jobs stop when you disconnect from the login node either by
choice or by internet connection problems. To keep a job alive you can
use a terminal multiplexer like `tmux`.

`tmux` allows you to run processes as usual in your standard bash shell

You start `tmux` on the login node before you get a interactive Slurm
session with `srun` and then do all the work in it. In case of a
disconnect you simply reconnect to the login node and attach to the `tmux`
session again by typing:
```
$ tmux attach
```
Or in case you have multiple session running:
```
$ tmux list-session
$ tmux attach -t SESSION_NUMBER
```

As long as the `tmux` session is not closed or terminated (e.g. by a
server restart) your session should continue. One problem with our
systems is that the `tmux` session is bound to the particular login server
you get connected to. So if you start a `tmux` session on login-1 on SAGA
and next time you get randomly connected to login-2 you first have to
connect to login-1 again by:
```
$ ssh login-1
```

To log out a `tmux` session without closing it you have to press Ctrl-B
(that the Ctrl key and simultaneously "b", which is the standard `tmux`
prefix) and then "d" (without the quotation marks). To close a session
just close the bash session with either Ctrl-D or type exit. You can get
a list of all `tmux` commands by Ctrl-B and the ? (question mark). See
also [this
page](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) for
a short tutorial of `tmux`. Otherwise working inside of a `tmux` session is
almost the same as a normal bash session.

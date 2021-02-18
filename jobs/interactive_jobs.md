

# Interactive jobs

Sometimes you might want to test or debug a calculation interactively,
but **running interactively on the login node is discouraged and not an
option**.

- [Asking for an interactive node](#asking-for-an-interactive-node)
- [Keeping interactive jobs alive](#keeping-interactive-jobs-alive)
- [Graphical user interfaces in interactive jobs](#graphical-user-interfaces-in-interactive-jobs)
- [The difference between salloc and srun](#the-difference-between-salloc-and-srun)


### Asking for an interactive node

Instead of running on a login node, you can ask the queue system to allocate
compute resources for you and once assigned, you can run the job(s)
interactively for as long as requested.  The examples below are for _devel_
jobs, but the procedure also holds for the [other job types
](choosing_job_types.md) except _optimist_ jobs.

On **Fram**:
```
$ srun --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount --pty bash -i
```

On **Saga**:
```
$ srun --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount --pty bash -i
```

When you are done, simply exit the shell (`exit`, `logout` or `^D`) to
end the job. 

The arguments between `srun` and `--pty` could be any arguments you
would have given to `sbatch` when submitting a non-interactive
job. However, `--qos=devel` is probably a good idea to avoid waiting
too long in the queue.

**Note that interactive jobs stop when you log out from the login
machine**, so unless you have very long days in office (or elsewhere,
for that matter), specifying more than 6-8 hrs runtime is not very
useful. An alternative is to start the job in a tmux session (next section).


### Keeping interactive jobs alive

Interactive jobs stop when you disconnect from the login node either by
choice or by internet connection problems. To keep a job alive you can
use a terminal multiplexer like tmux.

tmux allows you to run processes as usual in your standard bash shell

You start tmux on the login node before you get a interactive Slurm
session with `srun` and then do all the work in it. In case of a
disconnect you simply reconnect to the login node and attach to the tmux
session again by typing:
```
$ tmux attach
```
Or in case you have multiple session running:
```
$ tmux list-session
$ tmux attach -t SESSION_NUMBER
```

As long as the tmux session is not closed or terminated (e.g. by a
server restart) your session should continue. One problem with our
systems is that the tmux session is bound to the particular login server
you get connected to. So if you start a tmux session on login-1 on SAGA
and next time you get randomly connected to login-2 you first have to
connect to login-1 again by:
```
$ ssh login-1
```

On Fram the login nodes currently used are login-1-1 and login-1-2.

To log out a tmux session without closing it you have to press Ctrl-B
(that the Ctrl key and simultaneously "b", which is the standard tmux
prefix) and then "d" (without the quotation marks). To close a session
just close the bash session with either Ctrl-D or type exit. You can get
a list of all tmux commands by Ctrl-B and the ? (question mark). See
also [this
page](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) for
a short tutorial of tmux. Otherwise working inside of a tmux session is
almost the same as a normal bash session.


### Graphical user interfaces in interactive jobs

It is possible to run X commands, i.e., programs with a graphical user
interface (GUI), in interactive jobs. This allows you to get graphical output
back from your job running on a login node.

First, you must make sure that you have turned on *X forwarding* when logging
in to the cluster.  With `ssh` from a Linux or MacOS machine, you do this with
the `-Y` flag, e.g.:
```
$ ssh -Y fram.sigma2.no
```
or:
```
$ ssh -Y saga.sigma2.no
```

Check that the X forwarding works by running a graphical command like `emacs &`
and verify that it sets up a window.  (Note that due to network latency, it
can take a long time to set up a window.)

To be able to run X commands in interactive jobs, add the argument `--x11`
(note the lowercase `x`) to `srun`, like this:

On **Fram**:
```
$ srun --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount --x11 --pty bash -i
```

On **Saga**:
```
$ srun --ntasks=1 --mem-per-cpu=4G --time=00:30:00 --qos=devel --account=YourAccount --x11 --pty bash -i
```


### The difference between salloc and srun

An alternative to using `srun ... --pty bash -i` is to use
```
$ salloc --nodes=1 --time=00:30:00 --qos=devel --account=YourAccount
```

And equivalently on Saga.

As with `srun`, the arguments to `salloc` can be any you would have
given to `sbatch`. `salloc` will also give you an interactive shell, but note
that the shell **will be running on the login node**.  That means that you
*must* start all calculations with `srun` or `mpirun` or equivalent, to make
sure they run on the allocated compute node(s).  For this reason, **the srun
method is probably preferable in most cases** (it also works more like a batch
job).  However, there might be situations when the `srun` method doesn't work
properly, in which case you can try with `salloc`.

With this method, you can run X commands without adding any switch to the
`salloc` command, but note again that the commands will be run **on the login
node**.  In some cases, this is what you want (see for instance
:ref:`totalview_debugging`).

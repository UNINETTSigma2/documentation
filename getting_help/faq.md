# Frequently asked questions


## Access and connections

### How do I change my password?

Please consult {ref}`lost-passwords`.


### I forgot my password - what should I do to recover it?

Please consult {ref}`lost-passwords`.


### What is the ssh key fingerprint for our systems?

Please consult {ref}`this page <ssh>`.


### Connecting to the cluster

Typically users connect to our clusters with an SSH client. Please consult {ref}`this page <ssh>` for additional details.


### How can I access a compute node from the login node?

Log in to the login node, for instance Fram:
```console
$ ssh myusername@fram.sigma2.no
```

Then connect to the compute node (on Fram and Saga):
```console
$ ssh c3-5
```

Or on Betzy:
```console
$ ssh b4296
```

Notice that you typically can only log into a compute node where you have a running job.


### My ssh connections are freezing. How to fix it?

If your ssh connections more or less randomly are freezing, try
to add the following to `~/.ssh/config` file on your computer/laptop:
```cfg
ServerAliveCountMax 3
ServerAliveInterval 10
```

The above configuration is for [OpenSSH](https://www.openssh.com), if you're
using
[PUTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/docs.html)
you can take a look at this page explaining
[keepalives](https://the.earth.li/~sgtatham/putty/0.60/htmldoc/Chapter4.html#config-keepalive)
for a similar solution.

---

## Installing software

### I need to use Python but I am not satisfied with system default

You can choose different Python versions using either the {ref}`module-scheme` or
{ref}`Anaconda/Miniconda <installing-python-packages>`.
In Anaconda, you
typically load first the Anaconda module you like and then from within that you
can chose and configure the Python version and environment. Please consult the
[Anaconda documentation](https://docs.anaconda.com/) for details.

In cases where these routes still do not solve your problem or you would like
to install a package yourself, please consult this
page about {ref}`installing-software-as-user`.
If you are still stuck or would like
support, please contact the {ref}`support-line`.


### Can I install software as a normal user without sudo rights or a root account?

Yes. In fact, this is the recommended approach to install software that we do
not offer to all users.
Please consult this
page about {ref}`installing-software-as-user`.


---

## Compute and disk usage, in addition to allocated quota

### How can I check my disk quota and usage?

Please consult the page on {ref}`storage-quota`.


### How can I check my CPU hours quota and usage?

Please consult the page on {ref}`projects-accounting`.


---

## Graphical interfaces

### How can I export the display from a compute node to my desktop?

Please consult this note on {ref}`x11-forwarding`.

This example assumes that you are running an X-server on your local
desktop, which should be available for most users running Linux, Unix
and Mac Os X. If you are using Windows you must install some X-server
on your local PC.


---

## Jobs, submission, and queue system

### I am not able to submit jobs longer than the maximum set walltime

For all {ref}`job-types` there is a maximum walltime. If you try to set a 
walltime that is larger than this, the job will not be accepted when you submit it. We recommend you
to try to segment the job using {ref}`job-scripts`. If this does not suit your need,
please contact the {ref}`support-line`. The main
intention to have a limit on the max walltime is to make sure the queue system works as best as possible and
as such would give a better experience for most users.


### Where can I find an example of job script?

Here we have examples for {ref}`job-scripts-on-fram` and {ref}`job-scripts-on-saga`.


### When will my job start?

To find out approximately when the job scheduler thinks your job will
start, use the command:
```console
$ squeue --start -j <job_id>
```

where `<job_id>` is the number of the job you want to check.
This command will give you information about how many CPUs your job requires,
for how long, as well as when approximately it will start and complete.  It
must be emphasized that this is just a best guess, queued jobs may start
earlier because of running jobs that finishes before they hit the walltime
limit and jobs may start later than projected because new jobs are submitted
that get higher priority.


### How can I see the queue situation of my job(s)?

How can I see how my jobs are doing in the queue, if my jobs are idle, blocked, running etc. by issuing:
```console
$ squeue -u <username>
```
where `<username>` is your username. You can of course also check the queue by not adding a username. For additional
details on how to monitor job(s), please consult page about {ref}`monitoring-jobs`.

### Why are my devel/short/preproc jobs put in the “normal” queue even though I specify `--qos` in my job script?

The `--qos` specified jobs, like `devel`, `short` and `preproc`, by default run in the standard partition - i.e. `normal` but will have different properties. For detailed explanation see {ref}`queue-system`.
In order to see your jobs in the devel queue, use the following command, (you can replace `devel` with `short` or `preproc` to see the respective queues)
```console
$ squeue -q devel -u <username>
```

### Why does my job not start or give me error feedback when submitting?

Most often the reason a job is not starting is that the resources are busy. Typically there are many jobs waiting 
in the queue. But sometimes there is an error in the job script and you are asking for a configuration (say a combination of 
memory and cores) that is not possible. In such a cases you do not always get a message that the options are invalid on submission
and they might not be, but the combination will lead to a job that never starts.

To find out how to monitor your jobs and check their status see {ref}`monitoring-jobs`.

**Priority** means that resources are in principle available, but someone else has
higher priority in the queue. **Resources** means the at the moment the requested
resources are not available.


### How can I run many short tasks?

The overhead in the job start and cleanup makes it not practical to run
thousands of short tasks as individual jobs on our resources.

The queueing setup, or rather, the accounting system generates
overhead in the start and finish of a job. About a few seconds at each end
of the job for instance. This overhead is insignificant when running large parallel
jobs, but creates scaling issues when running a massive amount of
shorter jobs. One can consider a collection of independent tasks as one
large parallel job and the aforementioned overhead becomes the serial or
unparallelizable part of the job. This is because the queuing system can
only start and account one job at a time. This scaling problem is
described by [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law).

If the tasks are extremely short, you can use the example below. If you want to
spawn many jobs without polluting the queueing system, please use {ref}`array-jobs`.

By using some shell trickery one can spawn and load-balance multiple
independent task running in parallel within one node, just background
the tasks and poll to see when some task is finished until you spawn the
next:

```{eval-rst}
.. literalinclude:: ./files/multiple.sh
  :language: bash
```

And here is the `dowork.sh` script:

```{eval-rst}
.. literalinclude:: ./files/dowork.sh
  :language: bash
```

### Another user is clogging up the queue with lots of jobs!

The job scheduler on NRIS systems is normally configured to use a "Priority" attribute to determine which jobs to start next. This attribute increases over time (up to 7 days max), and is applied to a maximum of 10 jobs per user. There is no limit on the number of jobs or resources one user/project may request.

Superficially this may seem like a "first come first serve" system that allows a single user to 'block' others by submitting a large amount of jobs, but in reality it is a bit more complex since jobs may be of different sizes and lengths.

If there is a pending job with a high priority ranking that requires many CPUs for a long time, the scheduler will try to create a slot for this job in the future. As already running jobs finish up at different points in time, freeing up resources, the scheduler will attempt to squeeze in other jobs into the now-idle resource in a manner that does not extend the waiting time before the slot for the larger job is freed up in order to utilize the cluster as much as possible.

The "fairness" of this might be debatable, but in our experience this is the least unfair method that also ensures that the systems are idle as little as possible.

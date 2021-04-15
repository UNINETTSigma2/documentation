# Frequently Asked Questions

## Access and connections

### How do I change my password?

Please consult [this page](lost_forgotten_password,md).

### I forgot my password - what should I do to recover it?

Please consult [this page](lost_forgotten_password,md).

### What is the ssh key fingerprint for our systems?

Please consult [this page](../getting_help/create_ssh_keys.md)

### Connecting to the cluster

Typically users connect to our clusters with an SSH client. Please consult [this page](../getting_started/create_ssh_keys.md) for additional details.

### How can I access a compute node from the login node?

Log in to the login node, for instance Fram::

	ssh myusername@fram.sigma2.no

Then connect to the compute node (on Fram and Saga)::
	
	ssh c3-5

Or on Betzy::

	ssh b4296

Notice that you typically can only log into a compute node where you have a running job.

### My ssh connections are dying / freezing. How to fix it?

If your ssh connections more or less randomly are dying / freezing, try
to add the following to your *local* ``~/.ssh/config`` file::

    ServerAliveCountMax 3
    ServerAliveInterval 10

(*local* means that you need to make these changes to your computer,
not on our resources).

The above config is for `OpenSSH <https://www.openssh.org>`_, if you're
using
`PUTTY <https://www.chiark.greenend.org.uk/~sgtatham/putty/docs.html>`_
you can take a look at this page explaining
`keepalives <https://the.earth.li/~sgtatham/putty/0.60/htmldoc/Chapter4.html#config-keepalive>`_
for a similar solution.

## Installing software

### I need to use Python but I am not satisfied with system default

You can choose different Python versions using either the module system or Anaconda/Miniconda. See [here](../software/modulescheme.md) and [here](../software/userinstallsw/python.md) for the former. In Anaconda, you typically load first the Anaconda module you like and then from within that you can chose and configure the Python version and environment. Please consult the [Anaconda documentation](https://docs.anaconda.com/) for details.

In cases where these routes still does not solve your problem or you would like to install a package yourself, please consult this [page](../software/userinstallsw.md). If you are still stuck or would like support, please contact [support@metacenter.no](mailto:support@metacenter.no).

### Can I install software as a normal user without sudo rights or a root account?

Yes. In fact, this is the recommended approach to install software that we do not offer to all users. Please consult this [page](../software/userinstallsw.md).

## Compute and disk usage, in addition to allocated quota

### How can I check my disk quota and usage?

Please consult this [page](https://documentation.sigma2.no/files_storage/clusters.html?highlight=disk%20usage#usage-and-quota).


### How can I check my CPU hours quota and usage?

Please consult this [page](../jobs/projects_accounting.md).

## Other topics

### How can I export the display from a compute node to my desktop?

Please consult this [page](https://documentation.sigma2.no/getting_started/create_ssh_keys.html#x11-forwarding)

This example assumes that you are running an X-server on your local
desktop, which should be available for most users running Linux, Unix
and Mac Os X. If you are using Windows you must install some X-server
on your local PC.

## Jobs, submission and queue system

### I am not able to submit jobs longer than the maximum set walltime

For each [job type](../jobs/choosing_job_types.md) there is a maximum walltime. If you try to set a 
walltime that is larger than this, the job will not be accepted when you submit it. We recommend you
to try to segment the job using a [job script](../jobs/job_scripts.md). If this does not suit your need,
feel free to open a support ticket at [support@metacenter.no](support@metacenter.no). The main
intention to have a limit on the max walltime is to make sure the queue system works as best as possible and
as such would give a better experience for most users.

### Where can I find an example of job script?

You can find job script examples for Fram [here](https://documentation.sigma2.no/jobs/job_scripts/fram_job_scripts.html#job-scripts-on-fram) and for Saga [here](https://documentation.sigma2.no/jobs/job_scripts/saga_job_scripts.html#job-scripts-on-saga).

### When will my job start?

To find out approximately when the job scheduler thinks your job will
start, use the command::

	squeue --start -j <job_id>

where ``job_id`` is the id number of the job you want to check.
This command will give you information about how many CPUs your job requires,
for how long, as well as when approximately it will start and complete.  It
must be emphasized that this is just a best guess, queued jobs may start
earlier because of running jobs that finishes before they hit the walltime
limit and jobs may start later than projected because new jobs are submitted
that get higher priority.

### How can I see the queing situation of my job(s)?

How can I see how my jobs are doing in the queue, if my jobs are idle, blocked, running etc. by issuing::

	squeue -u <username>

where ``username`` is your username. You can of course also check the queue by not adding a username. For additional
details on how to monitor job(s), please consult this [page](../jobs/monitoring.md).

### Why does my job not start or give me error feedback when submitting?

Most often the reason a job is not starting is that the resources are busy. Typically there are many jobs waiting 
in the queue. But sometimes there is an error in the job script and you are asking for a configuration (say a combination of 
memory and cores) that is not possible. In such a cases you do not always get a message that the options are invalid on submission
and they might not be, but the combination will lead to a job that never starts.

To find out how to monitor your jobs and check their status see this [page](../jobs/monintoring.md).

Here follows a few typical gotchas:

**Priority vs. Resources**

Priority means that resources are in principle available, but someone else has higher priority in the queue. Resources means the at the moment the requested resources are not available.


### How can I customize emails that I get after a job has completed?

Use the mail command and you can customize it to your liking but make sure
that you send the email via the login node.

As an example, add and adapt the following line at the end of your script::

	echo "email content" | ssh stallo-1.local 'mail -s "Job finished: ${SLURM_JOBID}" <your_email_address>'

where ``your_email_address`` is your email address you want to be the receiver of the message.

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
described by `Amdahls Law <https://en.wikipedia.org/wiki/Amdahl's_law>`_.

If the tasks are extremly short, you can use the example below. If you want to
spawn many jobs without polluting the queueing system, please have a look [here](../jobs/job_scripts/array_jobs.md)

By using some shell trickery one can spawn and load-balance multiple
independent task running in parallel within one node, just background
the tasks and poll to see when some task is finished until you spawn the
next:

.. literalinclude:: ../files/multiple.sh
   :language: bash

And here is the ``dowork.sh`` script:

.. literalinclude:: ../files/dowork.sh
   :language: bash

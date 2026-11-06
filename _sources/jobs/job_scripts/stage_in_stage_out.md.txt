---
orphan: true
---

(stage-in-stage-out)=

# Staging In / Out Files from / to NIRD using Slurm

As an alternative to manually copying files from NIRD to the
{ref}`project work area <project-work-area>` on Olivia, one can
instruct the queue system to do the copying before a job is started.
This is called *staging in* files.  Similarly, one can instruct the
queue system to copy result files from the project work area to NIRD
after the job finishes (*staging out*).


## Staging In

Staging in of files is specified by adding one or more lines starting with `#STAGE
IN` to the job script, among the `#SBATCH` lines:
```
#STAGE IN /nird/datalake/NSxxxxK/some/path /cluster/work/project/nnxxxxk/another/path
```
Each line should contain one from-path and one to-path, in that order.
It is possible to stage in data from NIRD Datalake
(`/nird/datalake/NSxxxxK/...`) or NIRD Datapeak
(`/nird/datapeak/NSxxxxK/...`).  One can specify a single file, or a
directory, which will be copied recursively.  Note that wildcards
(like `*` or `?`) are *not* supported.

When the job has been submitted, the queue system will copy
`/nird/datalake/NSxxxxK/some/path` to
`/cluster/work/project/nnxxxxk/another/path`, etc, and will wait until
the files have been copied before starting the job.  While the file
copying is running, the job will be pending with reason
`BurstBufferStageIn`.

If the copying fails (for instance because the file or directory
doesn't exist), the job will be left pending, with the error message
from they copying as the job reason.  The job must then be cancelled,
and resubmitted after fixing the problem.


## Staging Out

Similarly, staging out is specified with lines starting with `#STAGE
OUT`:
```
#STAGE OUT /cluster/work/project/nnxxxxk/some/path /nird/datalake/NSxxxxK/another/path
```
To stage out to Datapeak, use `/nird/datapeak/NSxxxxK/...` instead of
`/nird/datalake/NSxxxxK/...` as the to-path.  Again, one can stage out
directories or single files.  Note that wildcards (like `*` or `?`)
are *not* supported.

When a job has finished, the files will be copied to NIRD.  While the
files are being copied, the job will be in state `completing`.  (It
also has a special Reason, but that is not visible with the default
output coloumns of `squeue`.)

If the copying fails (for instance because the file or directory
doesn't exist), the job will be left in state `STAGE_OUT` (`SO`), with
the error message from the copying as the job reason.  For instance:
```
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              5882    normal staging.      bhm SO       0:02      1 (burst_buffer/lua: JobHeldAdmin: slurm_bb_data_out failed: rsync: [sender] link_stat "/cluster/work/projects/nn9999k/bhm/nonexisting_file" failed: No such file or directory (2)
rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1336) [sender=3.2.7]
```

Jobs in this state must first be requeued before they can be
cancelled, like so:
```
$ scontrol requeue 5882
$ scancel 5882
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
```
(This only applies to jobs failing during stage-out.  Jobs failing
during stage-in can be cancelled directly.)


## Details

- Each `#STAGE IN` or `#STAGE OUT` line must contain exactly two
  paths: the from-path and the to-path, in that order, separated by
  white space.
- The paths must be absolute (start with `/`).
- Directory or file name with spaces in them must be quoted with `'`
  or `"`.  We do not recommend having spaces in file or directory
  names.
- Wildcards (like `*` or `?`) in the paths are *not* supported.
- Files are copied with `rsync`, so if the from-path ends with a `/`,
  the files and directories inside the from-path will be copied to
  to-path.  If the from-path does not end with a `/`, the directory
  (or file) will be created inside the to-path.
- Since files are copied with `rsync`, existing files in to-path will
  be overwritten if they exist in from-path.  This can for instance be
  used to refresh existing files with updated files.
- The files in from-path are *copied*, not *moved*.
- If a file copy fails, the files copied previously will not be
  deleted automatically.
- If a file copy fails during stage-in, the job can be cancelled as
  normal.  If it fails during stage-out, the job must first be
  requeued.


## Example

Here is a small example to illustrate how staging in/out works.  This
stages in input files from Datalake, and stages out an output file to
Datapeak.

The job script:
```
$ cat staging.sm
#!/bin/bash

# General job specifications:
#SBATCH -A nn9999k --mem-per-cpu=1G -t 10

# Stage in files:
#STAGE IN /nird/datalake/NS9999K/bhm/stagingtest/input_dir /cluster/work/projects/nn9999k/bhm
#STAGE IN /nird/datalake/NS9999K/bhm/stagingtest/input_file /cluster/work/projects/nn9999k/bhm

# Stage out files:
#STAGE OUT /cluster/work/projects/nn9999k/bhm/output_file /nird/datapeak/NS9999K/bhm/stagingtest

echo Doing some work...
sleep 42;
echo Storing results
echo 'Simulation results' > /cluster/work/projects/nn9999k/bhm/output_file
```

Before submitting the job, on one of the IO nodes (`svc[01-05]`):
```
$ ls -lR /nird/datalake/NS9999K/bhm/stagingtest/
/nird/datalake/NS9999K/bhm/stagingtest/:
total 1
drwxr-sr-x 2 bhm ns9999k 4096 Oct 27 12:47 input_dir/
-rw-r--r-- 1 bhm ns9999k    9 Oct 27 12:46 input_file

/nird/datalake/NS9999K/bhm/stagingtest/input_dir:
total 1
-rw-r--r-- 1 bhm ns9999k 7 Oct 27 12:47 another_input_file
$ ls -l /nird/datapeak/NS9999K/bhm/stagingtest/
total 0
$ ls -l /cluster/work/projects/nn9999k/bhm/
total 0
```
(So the datalake area contains input files, while the project work
area and the datapeak area are empty.)

Back on a login node, submit the job:
```
$ sbatch staging.sm
Submitted batch job 5869
```

While staging in files:
```
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              5869    normal staging.      bhm PD       0:00      1 (BurstBufferStageIn)
```

When the job has started running:
```
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              5869    normal staging.      bhm  R       0:11      1 c1-252
```

After job has finished, while staging out files:
```
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              5869    normal staging.      bhm CG       0:43      1 c1-252
```

On an IO node after the job has run, we see the files have been staged
in and out:
```
$ ls -l /cluster/work/projects/nn9999k/bhm
total 12
drwxr-sr-x 2 bhm ns9999k 4096 Oct 27 12:47 input_dir/
-rw-r--r-- 1 bhm ns9999k    9 Oct 27 12:46 input_file
-rw-r--r-- 1 bhm bhm       19 Oct 27 15:42 output_file
$ ls -l /nird/datapeak/NS9999K/bhm/stagingtest
total 1
-rw-r--r-- 1 bhm bhm 19 Oct 27 15:42 output_file
```

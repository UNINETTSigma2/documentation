# Array Jobs

To run many instances of the same job, use the `--array` switch to
`sbatch`.  This is useful if you have a lot of data-sets which you
want to process in the same way:

    sbatch --array=from-to [other sbatch switches] YourScript

You can also put the `--array` switch in an `#SBATCH` line inside the
script. _from_ and _to_ are the first and last task number.  Each
instance of `YourScript` can use the environment variable
`$SLURM_ARRAY_TASK_ID` for selecting which data set to use, etc.  (The
queue system calls the instances "array tasks".)  For instance:

    sbatch --array=1-100 MyScript

will run 100 instances of `MyScript`, setting the environment variable
`$SLURM_ARRAY_TASK_ID` to 1, 2, ..., 100 in turn.

It is possible to specify the task ids in other ways than `from-to`:
it can be a single number, a range (`from-to`), a range with a step
size (`from-to:step`), or a comma separated list of these. Finally,
adding `%max` at the end of the specification puts a limit on how many
tasks will be allowed to run at the same time. A couple of examples:

	Specification   Resulting SLURM_ARRAY_TASK_IDs
	1,4,42          # 1, 4, 42
	1-5             # 1, 2, 3, 4, 5
	0-10:2          # 0, 2, 4, 6, 8, 10
	32,56,100-200   # 32, 56, 100, 101, 102, ..., 200
	1-200%10        # 1, 2, ..., 200, but maximum 10 running at the same time

Note: spaces, decimal numbers or negative numbers are not allowed.

The instances of an array job are independent, they have their own
$SCRATCH and are treated like separate jobs.

To cancel all tasks of an array job, cancel the jobid that is returned
by `sbatch`.

A small, but complete example (for a _normal_ job on Saga):

```{eval-rst}
.. literalinclude:: files/minimal_array_job.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/minimal_array_job.sh`
```

Submit the script with `sbatch minimal_array_job.sh`.  This job will
process the datasets `dataset.1`, `dataset.2`, ..., `dataset.200` and
put the results in `result.1`, `result.2`, ..., `result.200`.

You can also find a more extended guide [here](job_array_howto.md).

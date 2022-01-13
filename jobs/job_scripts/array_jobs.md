---
orphan: true
---

(array-jobs)=

# Array Jobs

To run many instances of the same job, use the `--array` switch to `sbatch`.
This is useful if you have a lot of data-sets which you want to process in the
same way:

```console
$ sbatch --array=from-to [other sbatch switches] YourScript
```

You can also put the `--array` switch in an `#SBATCH` line inside the script.
_from_ and _to_ are the first and last task number. Each instance of
`YourScript` can use the environment variable `$SLURM_ARRAY_TASK_ID` for
selecting which data set to use, etc. (The queue system calls the instances
"array tasks".) For instance:

```console
$ sbatch --array=1-100 MyScript
```

will run 100 instances of `MyScript`, setting the environment variable
`$SLURM_ARRAY_TASK_ID` to 1, 2, ..., 100 in turn.

## Array job properties
### Specifying task IDs

It is possible to specify the task ids in other ways than `from-to`: it can be
a single number, a range (`from-to`), a range with a step size
(`from-to:step`), or a comma separated list of these. Finally, adding `%max` at
the end of the specification puts a limit on how many tasks will be allowed to
run at the same time. A couple of examples:

| Specification (`--array=`) | Resulting `SLURM_ARRAY_TASK_ID`s |
|----------------------------|----------------------------------|
| `1,4,42` | 1, 4, 42 |
| `1-5` | 1, 2, 3, 4, 5 |
| `0-10:2` | 0, 2, 4, 6, 8, 10 |
| `32,56,100-200` | 32, 56, 100, 101, 102, ..., 200 |
| `1-200%10` | 1, 2, ..., 200, but maximum 10 running at the same time |

```{note}
Spaces, decimal numbers or negative numbers are not allowed in the `--array`
specification.
```

### Array job resources

The instances of an array job are independent, they have their own `$SCRATCH`
({ref}`read more about storage locations here<storage-areas>`) and are treated
like separate jobs. Thus any resources request in the Slurm script is available
for each task.

### Canceling array jobs

To cancel all tasks of an array job, cancel the job ID that is returned by
`sbatch`. One can also cancel individual tasks with `scancel <array job
ID>:<task ID>`.

### Dependencies between array jobs

To handle dependencies between two or more array jobs one can use the
`--depend=aftercorr:<previous job ID>` (regular dependencies can also be used,
but we wanted to highlight this particular way since it can be beneficial with
array jobs), this will start the dependent array tasks as soon as the previous
corresponding array task has completed. E.g. if we start an array job with
`--array=1-5` and then start a second array job with `--array=1-5
--depend=aftercorr:<other job id>`, once task `X` of the first job is complete
the second job will start its task `X`, independently of the other task in the
first or second job.

## Example

A small, but complete example (for a `normal` job on Saga):

```{eval-rst}
.. literalinclude:: files/minimal_array_job.sh
  :language: bash
```

```{eval-rst}
:download:`minimal_array_job.sh <files/minimal_array_job.sh>`
```

Submit the script with `sbatch minimal_array_job.sh`. This job will process the
datasets `dataset.1`, `dataset.2`, ..., `dataset.200` and put the results in
`result.1`, `result.2`, ..., `result.200`. Each of the tasks will consist of
two processes (`--ntasks=2`) and get a total of `8GB` of memory (2 x
`--mem-per-cpu=4G`).

```{tip}
You can find a more extensive example {ref}`here <job-array-howto>`.
```

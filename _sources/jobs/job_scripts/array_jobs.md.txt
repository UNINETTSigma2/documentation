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

| Specification (`--array=`) | Resulting `SLURM_ARRAY_TASK_ID`s                        |
| -------------------------- | ------------------------------------------------------- |
| `1,4,42`                   | 1, 4, 42                                                |
| `1-5`                      | 1, 2, 3, 4, 5                                           |
| `0-10:2`                   | 0, 2, 4, 6, 8, 10                                       |
| `32,56,100-200`            | 32, 56, 100, 101, 102, ..., 200                         |
| `1-200%10`                 | 1, 2, ..., 200, but maximum 10 running at the same time |

```{note}
Spaces, decimal numbers or negative numbers are not allowed in the `--array`
specification.
```

The queue system allows job arrays with at most 1,000 array tasks, but
the maximal array task ID is 100,000 (thus `--array=900-1100` is allowed).

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
`result.1`, `result.2`, ..., `result.200`. **Note that your dataset files has to be named `dataset.1`, `dataset.2`, etc. for this example to work.** Make sure that the names of your dataset files and the names in your script are the same. Each of the tasks will consist of
two processes (`--ntasks=2`) and get a total of `8GB` of memory (2 x
`--mem-per-cpu=4G`).

If your files has inconsistent naming (for example "dataset_one", dataset_2", "my_dataset" etc.), you either have to rename your files or include code in your script to handle your files. Here is one way to handle inconsistent names:

```{warning}
You need to have the same number of files in your dataset directory as the number of tasks you specify in the `--array` switch i.e. count the number of files in your dataset directory and use that number in the `--array` switch. For example, to check how many csv files are in the directory named data, use `ls data/*.csv | wc -l` in the terminal.

```

```{code-block} bash
--------------
emphasize-lines: 5, 6, 7
--------------
#!/bin/bash
#SBATCH --account=YourProject
#SBATCH --time=1:0:0
#SBATCH --mem-per-cpu=4G --ntasks=2
#SBATCH --array=0-199             # we start at 0 instead of 1 for this
                                  # example, as the $SLURM_ARRAY_TASK_ID
                                  # variable starts at 0

set -o errexit # exit on errors
set -o nounset # treat unset variables as errors
module --quiet purge   # clear any inherited modules

DATASETS=(data/*)  # get all files in the directory named "data". Replace
                   # "data" with the path of your dataset directory.

FILE=${DATASETS[$SLURM_ARRAY_TASK_ID]}
FILENAME=$(basename ${FILE%.*})

YourProgram $FILE > ${FILENAME}.out
```

`DATASETS=(data/*)` will get all files in the directory named "data" and store them in an array. The array is indexed from 0, so the first file will be stored in `DATASETS[0]`, the second in `DATASETS[1]` and so on. The `SLURM_ARRAY_TASK_ID` variable is set by the Slurm system and is the task ID of the current task, with counting starting with 0.

```{tip}
If your datasets for example are csv files and the directory contains other file types, use DATASETS=(data/*.csv) instead.
```

Alternatively, you can save the names of you files in a text file and use the order of the filenames in the text file as an index. This is useful if you need the order of your files later or if you need to map the Slurm job output file to the correct dataset file.

Run for example these commands in the command line to create a text file with the names of your files:

```console
$ DATASETS=(data/*)
$ printf "%s\n" "${DATASETS[@]}" > map_files.txt
```

And use the following example as you run script:

```{code-block} bash
#!/bin/bash
#SBATCH --account=YourProject
#SBATCH --time=1:0:0
#SBATCH --mem-per-cpu=4G --ntasks=2
#SBATCH --array=0-199

set -o errexit # exit on errors
set -o nounset # treat unset variables as errors
module --quiet purge   # clear any inherited modules

IDX=($SLURM_ARRAY_TASK_ID)
FILE=$(sed "${IDX}q;d" map_files.txt)
FILENAME=$(basename ${FILE%.*})

YourProgram $FILE > ${FILENAME}.out
```

```{tip}
You can find a more extensive example {ref}`here <job-array-howto>`.
```

# Job Dependencies

In the following we demonstrate how to add dependecies between jobs using the SLURM option `--dependency`.
The full list of dependency types can be found in the [SLURM](https://slurm.schedmd.com/sbatch.html)
documentation, but we will show the most useful cases here:

| Option                 | Explanation                                                                     |
| :----------------------| :-------------------------------------------------------------------------------|
| `after:<jobid>`        | job can start after `<jobid>` has *started*                                     |
| `afterany:<jobid>`     | job can start after `<jobid>` has *completed* (any exit code)                   |
| `afterok:<jobid>`      | job can start only if `<jobid>` has *completed* with exit code 0 (success)      |
| `afternotok:<jobid>`   | job can start only if `<jobid>` has *completed* with exit code *not* 0 (failed) |

Several `<jobid>`s can be combined in a comma-separated list.

```{note}
The `--dependency` option must be added to the `sbatch` command *before* the name of the
job script, if you put it *after* the script it will be treated as an argument to the script, not
to the `sbatch` command. If the dependency was added successfully, you should see a `(Dependency)`
in the `NODELIST(REASON)` column of the `squeue` output.
```

#### Beware of exit status

With some of the options it is important to keep in mind the *exit status* of
your job script, to indicate whether or not the job finished successfully. By default the
script will return the exit status of the *last command* executed in the script, which in
general does not necessarily reflect the overall success of the job. It is then highly
recommended adding the following to the script:

```bash
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
```

as well as capturing errors in critical commands along the way:

```bash
mycommand || exit 1
```

and finally *explicitly* return 0 in case the script finishes successfully:

```bash
# Successful exit
exit 0
```

Standard SLURM errors like out-of-memory or time limit will of course be captured automatically.

#### Examples

- **Here `pre.sh` is a pre-processing step for `job-1.sh`, `job-2.sh`, etc:**
```bash
$ sbatch pre.sh
Submitted batch job 123123
$ for i in 1 2 3 4 5; do sbatch --dependency=afterok:123123 job-${i}.sh; done
Submitted batch job 123124
Submitted batch job 123125
Submitted batch job 123126
Submitted batch job 123127
Submitted batch job 123128
$ squeue -u $USER
       JOBID PARTITION    NAME     ST   USER   TIME  NODES NODELIST(REASON)
      123124    normal    job-1    PD     me   0:00      1 (Dependency)
      123125    normal    job-2    PD     me   0:00      1 (Dependency)
      123126    normal    job-3    PD     me   0:00      1 (Dependency)
      123127    normal    job-4    PD     me   0:00      1 (Dependency)
      123128    normal    job-5    PD     me   0:00      1 (Dependency)
      123123    normal    pre       R     me   0:28      1 c1-1
```


- **Here `post.sh` is a post-processing step for `job-1.sh`, `job-2.sh`, etc:**
```bash
$ for i in 1 2 3 4 5; do sbatch job-${i}.sh; done
Submitted batch job 123123
Submitted batch job 123124
Submitted batch job 123125
Submitted batch job 123126
Submitted batch job 123127
$ sbatch --dependency=afterok:123123,123124,123125,123126,123127 post.sh
Submitted batch job 123128
```

- **Here `job-2.sh` is a fallback/retry in case `job-1.sh` fails:**
```bash
$ sbatch job-1.sh
Submitted batch job 123123
$ sbatch --dependency=afternotok:123123 job-2.sh
Submitted batch job 123124
```

- **If for some reason you want your jobs to run one after the other:**

This is a bit cumbersome to do in a loop since the `sbatch` command returns the text string
"Submitted batch job" before showing the jobid, but we can extract it with a `awk '{ print $4 }'`
command (which returns the 4th entry in the string), and use it in a loop as follows (not that
the first job must be submitted individually, as it has no dependencies):

```bash
$ lastid=`sbatch job-1.sh | awk '{ print $4 }'`
$ echo $lastid
123123
$ for i in 2 3 4 5; do lastid=`sbatch --dependency=after:${lastid} job-${i}.sh | awk '{ print $4 }'`; echo ${lastid}; done
123124
123125
123126
123127
```

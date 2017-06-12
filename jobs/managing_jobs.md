Managing jobs
=============

The lifecycle of a job can be managed with as little as three different
commands:

1.  Submit the job with `sbatch <script_name>`.
2.  Check the job status with `squeue`. (to limit the display to only
    your jobs use `squeue -u <user_name>`.)
3.  (optional) Delete the job with `scancel <job_id>`.

You can also hold the start of a job:

**scontrol hold &lt;job\_id&gt;** Put a hold on the job. A job on hold will not start or block other
    jobs from starting until you release the hold.

**scontrol release &lt;job\_id&gt;** Release the hold on a job.



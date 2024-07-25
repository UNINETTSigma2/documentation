---
orphan: true
---

(squeue)=

# `squeue` output examples

Running `squeue -u UserName` will yield a table like the following:
```
             JOBID PARTITION    NAME      USER   ST       TIME  NODES NODELIST(REASON)
            120001    normal   job_1  UserName   PD       0:00      1 (Priority)
            120002    normal   job_2  UserName   PD       0:00      1 (Resources)
            120003    normal   job_3  UserName   PD       0:00      1 (Priority)
```
where:
- `JOBID` Is the ID given to the individual job when submitted to the queue. This ID can be used to gain more detailed information about the individual job with `seff` or .... <!-- TODO fill more -->
- `PARTITION` Is the type of job you are running. In this case it is `normal` , You can read more about these in <!-- reference to job type --> 
- `NAME` Is the name you have given your job when you submitted it to the queue.
- `ST` Is the state of the job. Right now all jobs are pending (`PD`). For a descrition of more job states, see {ref}`job-states`
- `TIME` Is the time elapsed since the jobs started running.
- `NODES` Number of nodes requested/running for the job.
- `NODELIST(REASON)` Describes different things in different states of the job execution lifetime depending on the state of the job. 
  - If the job is not currently running it will display a reason for this, some common ones are explained in {ref}`job-states`.
  - If the job is running it will display an indication of what nodes it is running in. This information is important in debugging issues if the job exited abnormally. We explain these further down


The nodelist will show a series of letters and numbers which index the nodes in which the different jobs are running. 
In FRAM and SAGA these will look like `cx-y`, 
while in betzy they will look like `bx` where for all machines both `x` and `y` are numbers describing the physical organizing of the machines
Following is an example of 
```
             JOBID PARTITION    NAME      USER   ST       TIME  NODES NODELIST(REASON)
            123001    normal   job_4  UserName    R      10:00      4 b[3379,3382,3387,3395]
            123002    normal   job_6  UserName    R      10:00      4 b[1334-1337]
```
In this example job `123001` is using four nodes, the indices of these nodes are listed in the `NODELIST` in square brackets separated by a coma. 
Job `123002` has a different nodelist description separated by `-`, this means that the nodes used are in a consecutive series. In this example `b[1334-1337]` means that nodes `b1334`, `b1335`, `b1336` and `b1337` are all used.

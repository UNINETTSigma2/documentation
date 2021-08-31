---
orphan: true
---

# schrodinger.hosts file example
```
# Schrodinger hosts file
#
# The hosts file consists of a series of entries, each describing a
# 'host' (machine) on which jobs can be run.  See the Installation
# Guide for a description of the settings that can be made here.
#
###############################  NOTE  ################################
# The 'localhost' entry is special:
# * Settings in the 'localhost' entry are implicitly included in
#   every other host entry as well, so settings common to all entries
#   can be placed in the localhost entry.
# * The 'schrodinger:', 'host:' and 'queue:' fields may not be used in
#   the localhost entry.
#######################################################################

# schrodinger.hosts for SAGA
name:        localhost
schrodinger: ${SCHRODINGER}
tmpdir:      /cluster/work/users/schrodinger
processors:  1

# 1 hour wall time, 40 tasks with default 1 cpu/task
name:        batch-small
host:        login-3
schrodinger: ${SCHRODINGER}
queue:       SLURM2.1
qargs:       --export=ALL --account=nnXXXXk --ntasks=40 --mem-per-cpu=3GB --time=01:00:00
processors:  40
tmpdir:      /cluster/work/users/schrodinger

# 7 days wall time 20 tasks with 4 cpus/task
name:        batch-long
host:        login-3
schrodinger: ${SCHRODINGER}
queue:       SLURM2.1
qargs:       --export=ALL --account=nnXXXXk --ntasks=20 --cpus-per-task=4 --mem-per-cpu=3GB --time=7-00:00:00
processors:  80
tmpdir:      /cluster/work/users/schrodinger

#
name:        batch-jaguar
host:        login-3
schrodinger: ${SCHRODINGER}
queue:       SLURM2.1
qargs:       --export=ALL --account=nnXXXXk --mem-per-cpu=1GB --ntasks=%TPP% --time=2-00:00:00
processors:  160
tmpdir:      /cluster/work/users/schrodinger
```
## 

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)
* [Hosts file keywords](host_file_settings.md)
* [Job control facility](job_control.md)
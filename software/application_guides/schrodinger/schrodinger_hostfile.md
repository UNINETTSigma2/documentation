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

#
name:        batch-small
host:        login-3
schrodinger: ${SCHRODINGER}
queue:       SLURM2.1
qargs:       --export=ALL --account=nnXXXXk --mem-per-cpu=1GB --time=01:00:00
processors:  40
tmpdir:      /cluster/work/users/schrodinger

#
name:        batch-long
host:        login-3
schrodinger: ${SCHRODINGER}
queue:       SLURM2.1
qargs:       --export=ALL --account=nnXXXXk --mem-per-cpu=1GB --time=7-00:00:00
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

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)
---
orphan: true
---

# The Hosts file
The Schrodinger Job Control facility obtains information about the hosts on which it will run jobs from the 
schrodinger.hosts file. This file contains many of the usual slurm parameters that you probably have seen before.
In addition, the schrodinger.hosts is used to set up the menus in the job settings dialog box in `Maestro`.

## How to obtain the schrodinger.hosts file
The schrodinger.hosts is located in `/cluster/software/Schrodinger` and  must be copied to your home to .schrodinger 
(`$USER/.schrodinger`). The .schrodinger directory in your home probably do not exist, so you need to create it first.

Copy and paste the following commands to get the hosts file:
1. `mkdir -p $USER/.schrodinger`
2. `cp /cluster/software/Schrodinger/schrodinger.hosts $USER/.schrodinger`

## Editing schrödinger.hosts
The first and most important thing to do is to add the correct account to your schrodinger.hosts.
Open `$USER/.schrodinger/schrodinger.hosts` with a text editor (for example `vim`) and edit the `qargs --account=` line.
<span style="background-color: #FFFF00">NOTE: Do not edit the localhost entry!</span>

[Example schrodinger.hosts file](schrodinger_hostfile.md)

In principle, you can edit the qargs to meet requirements of every type of job you want to submit (similar to a regular
slurm job script):
* `qargs:       --export=ALL --account=nnXXXXk --mem-per-cpu=1GB --time=04:00:00`

It is however more efficient to make several host entries with different settings depending on the jobs you are
  going to submit. 



[Keywords for schrodinger.hosts](host_file_settings.md)

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)
* [Hosts file keywords](host_file_settings.md)
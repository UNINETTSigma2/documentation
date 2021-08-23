---
orphan: true
---

# Using the Schrodinger suite

Load the desired Schrodinger suite on SAGA:
* `module purge`
* `module load Schrodinger/2021-2-intel-2020b`

Now you can call any Schrodinger software by using the \$SCHRODINGER variable from your command line, for example
`$SCHRODINGER/glide`.

You can also launch maestro by typing the command `maestro`. We however would generally encourage our users
to limit the use of graphical user interfaces to as little as possible. If you for some reason need to use the 
maestro gui, you must log in to SAGA with X11 forwarding, e.g. `ssh -Y you@saga.sigma2.no`.

To the extent it is possible, we recommend preparing input files etc. using a local version of maestro and uploading
the files to SAGA (`scp -r input_files/ you@saga.sigma2.no:/some/existing/directory`). Jobs can then be submitted from
the command line using the `$SCRODINGER` variable. For example:

* `"${SCHRODINGER}/glide" glide-grid_1.in -OVERWRITE -HOST batch-small -TMPLAUNCHDIR`

The above command submits the pre-created input file glide-grid_1.in. The `-OVERWRITE -HOST batch-small` tells 
Schrodinger to you use the job settings and qargs defined in your local [schrodinger.hosts](schrodinger_hosts.md) file 
with entry name batch-small. 

##

### Go to:
* [Schrodinger main page](schrodinger.md)
* [Using the Schrodinger suite](schrodinger_usage.md)
* [Setting up the Hosts file](schrodinger_hosts.md)
* [Hosts file keywords](host_file_settings.md)

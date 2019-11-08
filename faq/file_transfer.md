# File transfer

Access to the systems (Fram, Saga, NIRD) is permitted only trough SSH.
One can use `scp` or `sftp` to upload or download data.  There are
graphical frontends for these tools, for instance [WinSCP](WinSCP.md)
for MS Windows.

To transfer files between an HPC cluster (Fram or Saga) and NIRD, one can use regular
`cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas, since these are
mounted on the cluster login nodes.  For more information,
please check out the [Fram and Saga](../clusters.md) storage page.

## Basic tools (scp, sftp)

* `scp` - secure copy files between hosts on a network

```
# copy single file to home folder on Fram
# note that folder is ommitted, home folder being default
scp my_file.tar.gz <username>@fram.sigma2.no:

# copy a directory to work area
scp -r my_dir/ <username>@fram.sigma2.no:/cluster/work/users/<username>/
```

* `sftp` - interactive secure file transfer program (Secure FTP)

```
# copy all logs named starting with "out" from project1 folder to
# /nird/projects/project1 on NIRD
sftp <username>@nird.sigma2.no
sftp> lcd project1
sftp> cd /nird/projects/project1
sftp> put out*.log
```

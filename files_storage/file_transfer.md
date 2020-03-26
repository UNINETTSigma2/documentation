# File transfer

Access to the systems (Betzy, Fram, Saga, NIRD) is permitted only trough [secure
shell](https://en.wikipedia.org/wiki/Secure_Shell) (ssh).  One can use `scp`
(secure copy) or `sftp` (secure file transfer protocol) to upload or download
data. There are graphical frontends for these tools, for instance
[WinSCP](WinSCP.md) for MS Windows.

To transfer files between an HPC cluster (Betzy, Fram or Saga) and NIRD, one can use regular
`cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas, since these are
mounted on the cluster login nodes.  For more information,
please check out the [Betzy, Fram and Saga](clusters.md) storage page.


### scp: secure copy between hosts on a network

```bash
# copy single file to home folder on Fram
# note that folder is ommitted, home folder being default
scp my_file.tar.gz <username>@fram.sigma2.no:

# copy a directory to work area
scp -r my_dir/ <username>@fram.sigma2.no:/cluster/work/users/<username>/

# copy a result directory back to my laptop
scp -r <username>@fram.sigma2.no:/cluster/work/users/<username>/my_results /home/some/place
```


### sftp: interactive secure file transfer program (Secure FTP)

```bash
# copy all logs named starting with "out" from project1 folder to
# /nird/projects/project1 on NIRD
sftp <username>@nird.sigma2.no
sftp> lcd project1
sftp> cd /nird/projects/project1
sftp> put out*.log
```

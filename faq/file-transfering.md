# File transfer to/from Fram

Access to Fram is permitted only trough SSH.
One can use *scp* and *sftp* to upload or download data from Fram.

To transfer files between Fram and NIRD, regular *cp*/*mv* commands can be
used, since NIRD file systems are mounted on Fram. For more information,
please check out the [NIRD documentation](nird.md).

## Basic tools (scp, sftp)

* scp - secure copy files between hosts on a network

```
# copy single file to home folder on Fram
# note that folder is ommitted, home folder being default
scp my_file.tar.gz <username>@fram.sigma2.no:

# copy a directory to work area
scp -r my_dir/ <username>@fram.sigma2.no:/cluster/work/users/<username>/
```

* sftp - interactive secure file transfer program (Secure FTP)

```
# copy all logs named starting with "out" from project1 folder to
# /nird/projects/project1
sftp <username>@fram.sigma2.no
sftp> lcd project1
sftp> cd /nird/projects/project1
sftp> put out*.log
```

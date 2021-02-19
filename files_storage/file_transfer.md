# File transfer

Access to the compute and storage systems (Betzy, Fram, Saga, NIRD) is permitted only trough [secure
shell](https://en.wikipedia.org/wiki/Secure_Shell) (ssh).

One can use `scp`
(secure copy) or `sftp` (secure file transfer protocol) and `rsync` to upload or download
data. There are graphical frontends for these tools, for instance
[WinSCP](file_transfer/WinSCP.md) for MS Windows.


## Transferring files between Betzy/Fram/Saga and NIRD

Since NIRD is mounted on the login nodes of Betzy, Fram, and Saga,
one can use regular
`cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas.

For more information, please check out the [Betzy, Fram and Saga](clusters.md)
storage page.

### scp: secure copy between hosts on a network

Few typical examples when using `scp`:

```bash
# copy single file to home folder on Fram
# note that folder is ommitted, home folder being default
$ scp my_file.tar.gz <username>@fram.sigma2.no:

# copy a directory to work area
$ scp -r my_dir/ <username>@fram.sigma2.no:/cluster/work/users/<username>/

# copy a result directory back to my laptop
$ scp -r <username>@fram.sigma2.no:/cluster/work/users/<username>/my_results /home/some/place
```


### sftp: interactive secure file transfer program (Secure FTP)

In this example we copy all logs with names that start with "out" and
end with ".log" from the `project1` folder to `/nird/projects/project1` on NIRD:

```bash
$ sftp <username>@nird.sigma2.no

sftp> lcd project1
sftp> cd /nird/projects/project1
sftp> put out*.log
```


## Sharing files with others

You may can use [FileSender](https://filesender.org/) to share files with
others.
We have two instances running:

* https://filesender2.uio.no/
* https://filesender.uninett.no/

Please use one of these: we can give support if needed.

If the data you want to share could be useful for the scientific community,
please use the [NIRD Archive](../nird_archive/user-guide.md).

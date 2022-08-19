(file-transfer)=

# File transfer

Access to the compute and storage systems (Betzy, Fram, Saga, NIRD) is permitted only trough [secure
shell](https://en.wikipedia.org/wiki/Secure_Shell) (ssh).

One can use `scp`
(secure copy) or `sftp` (secure file transfer protocol) and `rsync` to upload or download
data. There are graphical front-ends for these tools, for instance
[WinSCP](file_transfer/WinSCP.md) for MS Windows.


## Transferring files between Betzy/Fram/Saga and NIRD

Since NIRD is mounted on the login nodes of Betzy, Fram, and Saga,
one can use regular
`cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas.

For more information, please check out the [Betzy, Fram, and Saga](clusters.md)
storage page.


## Transferring files between your computer and a cluster using scp

Few typical examples when using `scp`:

Copy single file to home folder on Fram.
Note that if the folder is omitted (the part after the `:`), the home folder is used by default:
```console
$ scp my_file.tar.gz <username>@fram.sigma2.no:
```

Copy a directory to work area:
```console
$ scp -r my_dir/ <username>@fram.sigma2.no:/cluster/work/users/<username>/
```

Copy a result directory back to my laptop:
```console
$ scp -r <username>@fram.sigma2.no:/cluster/work/users/<username>/my_results /home/some/place
```


## Interactive secure file transfer program (Secure FTP)

In this example we copy all logs with names that start with "out" and
end with ".log" from the `project1` folder to `/nird/projects/project1` on NIRD:

```console
$ sftp <username>@nird.sigma2.no

sftp> lcd project1
sftp> cd /nird/projects/project1
sftp> put out*.log
```

### Troubleshooting - Broken pipe error during `scp`/`sftp`/`rsync`

The organization which provides the network to the clusters, may perform daily
housekeeping of their DNS and then the connection from outside to the NRIS
services can drop.  This can cause a "broken pipe" error during file transfer
from outside. To avoid this, especially while copying large datasets, it is
recommended to open a `screen` session from the NRIS cluster/storage  login
node and to pull/push the data from there.


## Sharing files with others

You can use [FileSender](https://filesender.uninett.no/) to share files with
others.

There are two instances of this service:
- <https://filesender2.uio.no/>
- <https://filesender.uninett.no/>

If the data you want to share could be useful for the scientific community,
please use the [NIRD Archive](../nird_archive/user-guide.md).

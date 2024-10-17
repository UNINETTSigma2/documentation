(file-transfer)=

# File transfer

```{admonition} Summary: use rsync for file transfer

For file transfer to/from and between compute and storage systems (Betzy, Fram,
Saga, NIRD), **we recommend `rsync`**. This tool is often faster than `scp` (for
many small files and it does not copy files that are already there) and
potentially also safer against accidental file overwrites.
For more details, see {ref}`advantages-over-scp`.

When using `rsync`, there is **no need to zip/tar files first**.

On Windows, many other tools exist ([WinSCP](https://winscp.net/),
[FileZilla](https://filezilla-project.org/),
[MobaXterm](https://mobaxterm.mobatek.net/), and others), but we recommend to
use `rsync` through [Windows Subsystem for Linux
(WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).
```


## Transferring files between your computer and a compute cluster or storage resource

This is a good starting point but below we will explain what these components
and options mean:
```console
$ rsync --info=progress2 -a file-name       username@cluster:receiving-directory
$ rsync --info=progress2 -a directory-name/ username@cluster:receiving-directory/directory-name
```

- `--info=progress2`: This will show progress (how many percent, how much time
  left). You can also leave it out if you don't need to know how far the
  copying is. There is also a `--progress` option but that one will show
  progress for each file individually and often you rather want to know the
  overall progress.
- `-a`: Preserves ownership and time stamp and includes the `-r` option which copies
  folders recursively.
- `file-name` or `directory-name`: These are on your computer and you want to
  transfer them to the receiving server.
- `username`: Your username on the remote cluster. If your usernames on your
  local computer and on the remote resource are the same, you can leave out the
  `username@` part.
- `cluster`: The remote server. For example: `saga.sigma2.no`.
- `receiving-directory`: The directory on the remote server which will receive the file(s) and/or directories.

If you want to make sure that `rsync` does not overwrite files that are newer
on the receiving end, add the `--update` option.

If you want to `rsync` between two computers that both offer an SSH connection, note that then
you can use `rsync` both ways: from cluster A to cluster B, but also the reverse.

````{admonition} rsync directory
Please note that there is a trailing slash (`/`) at the end of the first argument in the 
syntax of the second command, while rsync directories, ie:

```console
$ rsync --info=progress2 -a directory-name/ username@cluster:receiving-directory/directory-name
```
This trailing slash (`/`) signifies the contents of the directory `directory-name`. 
The outcome would create a hierarchy like the following on your cluster:
```console
~/receiving-directory/directory-name/contents-of-the-dir
```

Without the trailing slash, `directory-name`, including the directory, would be placed within your receiving directory.
The outcome would be the following on the cluster:
```console
~/receiving-directory/directory-name/directory-name/contents-of-the-dir
```
````


## rsync using compression

If you have a strong CPU at both ends of the line, and you’re on a slow
network, you can save bandwidth by compressing the data with the `-z` flag:

```console
$ rsync --info=progress2 -az file-name      username@cluster:receiving-directory
$ rsync --info=progress2 -az directory-name username@cluster:receiving-directory/directory-name
```


## Problem with many small files

Many small files are often not great for the transfer (although `rsync` does
not seem to mind but for `scp` this can make a big difference, see below). Many
tiny files are often also a problem for parallel file systems. If you develop
programs for high-performance computing, avoid using very many tiny files.


(advantages-over-scp)=

## Advantages over scp and similar tools

- `rsync` will not transfer files if they already exist and do not differ.
- With `rsync --update` you can avoid accidentally overwriting newer files in the destination directory.
- You can use compression for file transfer.
- Resumes interrupted transfers.
- More flexibility and better cross-platform support.

Typically people recommend `scp` for file transfer and we have also done this
in the past. But let us here compare `scp` with `rsync`.  In this example I
tried to transfer a 100 MB file from my home computer (not on the fast
university network) to a cluster, either as one large file or split into 5000
smaller files.

For one or few files it does not matter:
```bash
$ scp file.txt username@cluster:directory
# 81 sec

$ rsync --info=progress2 -a file.txt username@cluster:directory
# 79 sec

$ rsync --info=progress2 -az file.txt username@cluster:directory
# 61 sec
```

However, **it can matter a lot if you want to transfer many small files**.
Notice how the transfer takes 10 times longer with `scp`:
```{code-block} bash
---
emphasize-lines: 2, 5
---
$ scp -r many-files username@cluster:directory
# 833 sec

$ rsync --info=progress2 -a many-files username@cluster:directory/many-files
# 81 sec

$ rsync --info=progress2 -az many-files username@cluster:directory/many-files
# 62 sec
```

In the above example, `scp` struggles with many small files but `rsync` does
not seem to mind.  For `scp` we would have to first `tar`/`zip` the small files
to one large file but for `rsync` we don't have to.

````{admonition} How was the test data created?
Just in case anybody wants to try the above example on their own, we used this
script to generate the example data:
```bash
#!/usr/bin/env bash

# create a file that is 100 MB large
base64 /dev/urandom | head -c 100000000 > file.txt

# split into 5000 smaller files
mkdir -p many-files
cd many-files
split -n 5000 ../file.txt
```
````


## Transferring files between Betzy/Fram/Saga and NIRD

Since NIRD is mounted on the login nodes of Betzy, Fram, and Saga,
one can use regular
`cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas.

For more information, please check out the page about
{ref}`storage-areas`.


## What to do if rsync is not fast enough?

Disk speed, meta-data performance, network speed, and firewall speed may limit
the transfer bandwidth.

If you have access to a network with a large bandwidth and you are sure that
you are limited by the one `rsync` process and not by something else, you can
start multiple `rsync` processes, by piping a list of paths to `xargs` or
`parallel` which launches multiple `rsync` instances in parallel. But please
mind that this way you can saturate the network bandwidth for other users and
also saturate the login node with `rsync` processes or overwhelm the file
system. If you have to transfer large amount of data and one `rsync` process is
not enough, we recommend that you talk to us first: {ref}`support-line`.

Please also **plan for it**: If you need to transfer large amount of data,
don't start on the last day of your project. Data transfer may take hours or
even days.


## Troubleshooting: "Broken pipe" error during transfer

The organization which provides the network to the clusters, may perform daily
housekeeping of their [DNS](https://en.wikipedia.org/wiki/Domain_Name_System)
and then the connection from outside to the NRIS services can drop. This can
cause a "broken pipe" error during file transfer from outside.

One way to avoid this, especially while copying large datasets, is to use IP
addresses instead of domain names.

One way to get the IP of one of the login nodes (example: Saga):
```console
$ nslookup saga.sigma2.no
```

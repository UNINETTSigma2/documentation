(storage-quota)=

# Storage quota

```{contents} Table of Contents
```

```{warning}
**Frequently asked questions**

- **I cannot copy files although we haven't used up all space**:
  You have probably exceeded the quota on the number of files.

- **I have moved files to the project folder but my home quota usage did not go down**:
  Moving files does not change ownership of the files. You need to also change the ownership of the files
  in the project folder from you to the group (change the ownership from 'username_g' to 'username'; see also below).
```


## What is quota and why is it needed?

Storage is a shared and limited resource and in a number of places we need to
enforce quota to avoid that some script accidentally fills up the disk and the
system becomes unusable for everybody.

Storage quota is specified in number of files (or "inodes" or file chunks) and
space. The former limits how many files you or a group may own. When this limit
is reached, you or the group cannot create new files (but you might still
increase the size of existing files). The space limit affects the aggregated
size of all your files or files of a group. When this limit is reached you or
the group cannot store more data (new data or increasing file sizes) on the
system.

Often it is intended that storage quota applies to a specific folder on the
file system. For example, the so-called HOME quota shall apply to your home
folder and its location is often stored in the environment variable `$HOME`.
Another example, a project may have dedicated quota for data stored under their
project folder which is found under `/cluster/projects/nnABCDk` where `nnABCDk`
is the account name of your project. Because file systems have different
features, unfortunately it is not always guaranteed that what you observe on
the system matches this intention. Below, we will discuss how to detect and
troubleshoot such situations.


## Getting information about your usage and quota

We can get an overview with the `dusage` command.

`dusage` is not a built-in Unix command but rather a tool which we have
developed for NRIS clusters to wrap around lower-level commands and tools to
get a quick overview. The actual output might be different for every user:
```console
$ dusage

dusage v0.2.0
                     path    backup    space used     quota      files       quota
-------------------------  --------  ------------  --------  ---------  ----------
                 /cluster        no       6.4 GiB         -     17 509           -
       /cluster/home/user       yes       4.9 GiB  20.0 GiB     17 442     100 000
 /cluster/work/users/user        no      12.0 KiB         -          3           -
/cluster/projects/nnABCDk       yes      12.0 KiB  10.0 TiB          3   1 000 000
/cluster/projects/nnABCDk       yes       0.0 KiB   1.0 TiB          0   1 000 000
/cluster/projects/nnABCDk       yes       1.5 TiB   2.0 TiB    359 663   2 000 000
/cluster/projects/nnABCDk       yes       2.8 TiB  10.0 TiB  1 889 937  10 000 000

- This script is still being tested, unsure whether the backup information is correct.
  Please report issues at https://github.com/NordicHPC/dusage.
```


## Troubleshooting: Disk quota is full

- **Why this can be surprising for users and difficult to debug for staff**:
  - On Saga and Fram: Depending on the state of the file system there can be a
    lag between going over quota and experiencing "Disk quota exceeded" errors.
  - On Saga and Fram: If you moved files and kept wrong group permissions, this
    can exceed quota but we have overnight scripts which fix group permissions
    so it can look good again next morning.
  - `dusage` can indicate that you are above quota although `du` may show that
    there are almost no files or data used: the reason is that moving files
    does not change ownership and in this case `du` and `dusage` can give a different
    information. Only `dusage` gives you reliable information about how your
    quota is affected.

- **Recovery on Fram and Saga**:
  - Moving files to project data or `$USERWORK` may not be enough since `mv`
    preserves group permissions. Therefore you have the following options:
    - Copy files and then carefully delete the files in `$HOME`.
    - Move files and adjust group permission with `chown` or `chgrp`.
    - Move files and wait overnight for our scripts to adjust them for you.

- **Recovery on Betzy**:
  - Try to move data from `$HOME` to project data.
  - Consider using `/cluster/work/users/$USER` (`$USERWORK`).
  - If the above are not enough or not suitable, contact support and discuss
    whether it can make sense to increase project or user quota.

- **Recommendations**:
  - If you tend to fill up quota in your job scripts, add a `dusage` at the
    beginning and at the end of the job script. Having the output will make
    diagnostics easier. If you don't `dusage` right when you run the job, then
    a job crash and a later `dusage` may tell different stories.
  - `rsync` users: Please be careful adjusting the group ownership on Saga and
    Fram.


## Troubleshooting: Too many files/inodes on Fram

Fram has an inode quota for `/cluster`:
```{code-block}
---
emphasize-lines: 3
---
                     path    backup    space used    quota (s)    quota (h)    files    quota (s)    quota (h)
-------------------------  --------  ------------  -----------  -----------  -------  -----------  -----------
                 /cluster        no     313.7 GiB            -            -   50 442    1 000 000    3 000 000
       /cluster/home/user       yes       1.7 GiB     20.0 GiB     30.0 GiB   45 838      100 000      120 000
 /cluster/work/users/user        no     243.8 GiB            -            -    3 764            -            -
/cluster/projects/nnABCDk       yes       4.0 KiB      1.0 TiB      1.1 TiB        1    1 048 576    1 150 976
/cluster/projects/nnABCDk       yes     178.1 GiB      1.0 TiB      1.1 TiB   12 816    1 048 576    1 150 976
/cluster/projects/nnABCDk       yes       9.8 TiB     10.0 TiB     10.0 TiB  571 440   10 000 000   11 000 000
```

We can think of "inodes" as files or file chunks.

This means that on Fram it is possible to fill the "files"/inode quota by
putting more than 1 M files in `/cluster/work/users/user` although the latter
is not size-quota controlled.

Please contact support if you are in this situation and we can then together evaluate
whether it makes sense to increase the `inode` quota for you.


## Troubleshooting: Too many files in a Conda installation

- A Conda installation can fill your storage quota because it can install
  thousands of files.
- **Recommendation**: Do not install Conda environment into `$HOME`
- **Recovery** from a `$HOME`-installed Conda environment:
  - Install a new environment into project data or `$USERWORK` and then delete
    the `$HOME`-installed Conda environment.
  - Advanced alternative: Use a Singularity container for the Conda environment.


## Changing file ownership on Fram or Saga

This section is not relevant for Betzy as disk quotas on Betzy are based on
directories instead of groups.

Since file permissions are persistent across the file system, it might be
necessary to manually change the ownership of one or more files.  This page
will show an example of how to change ownership on a file that was moved from
`$HOME` to `$USERWORK` in order to update the disk quotas.

In this example we have a file in our `$HOME` called "myfile.txt" which is 1
GiB in size that we're moving to `$USERWORK` for use in a job:

```console
$ ls -l

total 1048576
-rw-rw-r-- 1 username username_g 1073741824 Nov 13 13:11 myfile.txt
```

```console
$ mv myfile.txt /cluster/work/users/username
```

By checking our disk usage with `dusage` we can see that the file is still
counted towards the `$HOME` quota. The reason for this is that the file is
still owned by the `username_g` group, which is used for the `$HOME` quota.:

Files in `$USERWORK` should be owned by the default user group, in this - the
group named `username`.  To change the file group ownership we can use the
command `chgrp`:
```console
$ chgrp username myfile.txt
```

```console
$ ls -l

total 1048576
-rw-rw-r-- 1 username username 1073741824 Nov 13 13:11 myfile.txt
```

The file is now owned by the correct group and we can verify that the disk
quotas have been updated by running `dusage` again.

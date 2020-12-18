# Backup on Betzy, Fram, Saga, and NIRD

```{warning}
**Only home directories with enforced quotas are backed up**

Any \$HOME area using more then 20 GiB or more than 100000 files is
not backed up. To have your \$HOME backed up, you need to shrink the
disk usage below the 20 GiB limit *and* below 100000 files *and*
[notify support](/getting_help/support_line.md).

If "dusage" repots 0 Bytes limits, this means that you **do not have disk quotas**
activated and this means that these folders **are not backed up**.
```


## Betzy, Fram and Saga

Home directories on Betzy, Fram and Saga (`/cluster/home/$USER`) are backed up
nightly to user's home directory on NIRD
(`/nird/home/$USER/backup/fram` or `/nird/home/$USER/backup/saga`).

`$HOME/nobackup` and `$HOME/tmp` directories are excluded from the backup.

The `nnXXXXk` project directories (`/cluster/projects/nnXXXXk`) are backed up
nightly to NIRD.

The shared areas in `/cluster/shared` **are not backed up** to NIRD.

The scratch areas in `/cluster/work/jobs` and `/cluster/work/users` **do not have any backup**.


## NIRD

Both home directories (`/nird/home/$USER`) and project areas (`/nird/projects/NSxxxxK`) have
backup in the form of snapshots and geo-replication.

Snapshots are taken with the following frequencies:
* `/nird/home/$HOME`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
  - this also includes snapshots of the Betzy, Fram and Saga home directory backup
* `/nird/projects/NSxxxxK`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

Snapshots are available at the following places:
- `/nird/home/u1/.snapshots`
- `/nird/projects/NSxxxxK/.snapshots`


## How to recover deleted/overwritten data

To recover a deleted or overwritten file `/nird/projects/NSxxxxK/dataset1/myfile`,
you can copy a snapshot back to the folder and restore the deleted/overwritten file:

```
$ cp /nird/projects/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/NSxxxxK/dataset1/
```

Select the DATE accordingly to your case.

Similarly, a deleted/overwritten file in the home directory can be recovered like this:

```
$ cp /nird/home/u1/.snapshots/DATE/$USER/mydir/myfile /nird/home/$USER/mydir/
```

Note that snapshots are taken every night only. This means that deleted files
which did not exist yet yesterday cannot be recovered from snapshots.

To recover a deleted or overwritten file in `/cluster/home/` on either Betzy Fram or Saga
go to your home directory on NIRD, go to the backup folder and then browse in the directory
corresponding to the hpc system you come from for the file you want to restore.
You can then use scp/sftp to transfer a copy to your home directory on the hpc system.

To recover a deleted or overwritten file in `/cluster/projects/nnxxxxk`
please contact support@metacenter.no and specify the name of the hpc system, the nn project
and the file (or folder).

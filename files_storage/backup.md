(storage-backup)=

# Backup on Betzy, Fram, Saga and NIRD

```{warning}
**Only home and project folders with enforced quotas are backed up**

Any `$HOME` area using more than 20 GiB or more than 100 000 files is
not backed up. To have your `$HOME` backed up, you need to shrink the
disk usage below the 20 GiB limit *and* below 100 000 files *and*
[notify support](/getting_help/support_line.md).

If `dusage` reports that no limits are set, this means that you **do not have disk quotas**
activated and this means that these folders **are not backed up**.
```


## Betzy, Fram and Saga

The storage backup on Betzy, Fram and Saga happens in two steps: first as a nightly backup from the
respective machines over to NIRD, and then NIRD is backed up as described in the NIRD section below.

The following areas on Betzy, Fram and Saga are backed up nightly to NIRD:
- `/cluster/home`, excluding `$HOME/nobackup` and `$HOME/tmp`
- `/cluster/projects`

The following areas are **not backed up**:
- `/cluster/share`
- `/cluster/work`

The work area `/cluster/work` also enforces an automatic cleanup strategy, and
is **not** meant for permanent storage.
Files in this area will be **deleted** after 42 or 21 days, depending on the storage capacity,
see [User work area](user-work-area) for details.


### Where the backups are located

**HOME**: The nightly backups of the `$HOME` areas on Betzy, Fram and Saga end up in your `$HOME` area on
NIRD, which is geo-replicated between NIRD-TOS (Tromsø) and NIRD-TRD (Trondheim):
- `/nird/home/$USER/backups/betzy`
- `/nird/home/$USER/backups/fram`
- `/nird/home/$USER/backups/saga`

To recover a deleted or overwritten file in `/cluster/home/$USER` on either Betzy, Fram or Saga
go to your home directory on NIRD, go to the backup folder and then browse in the directory
corresponding to the HPC system you come from for the file you want to restore.
You can then use `rsync` to transfer a copy to your home directory on the HPC system
(see also our guide about {ref}`file-transfer`).
If you have difficulty accessing NIRD, please contact support.

**PROJECTS on Fram and Betzy**: The nnXXXXk project areas on Betzy and Fram
are backed up to Saga:
- `/cluster/backup/betzy/projects/nnXXXXk`
- `/cluster/backup/fram/projects/nnXXXXk`

**PROJECTS on Saga**: The nnXXXXk project areas on Saga
are backed up to Betzy:
- `/cluster/backup/saga/projects/nnXXXXk`

If you have project access to the cluster where backups are stored, then you
can retrieve them yourself. If you cannot access the cluster that holds the
project backups, please contact support and we will help you restoring your
data.


## NIRD

Both home directories (`/nird/home/$USER`) and project areas (`/nird/projects/NSxxxxK`) have
backup in the form of snapshots and geo-replication (only NS projects are geo-replicated,
not the NN project backups from Betzy, Fram and Saga mentioned above).

Snapshots are taken with the following frequencies:
* `/nird/home/$USER`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
  - this also includes the nightly backup of the Betzy, Fram and Saga home directory as described above
* `/nird/projects/NSxxxxK`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
* `/tos-project4/fram`, `/trd-project4/saga`, `/trd-project4/betzy`:
  - site dependent, but daily for a few days and weekly for a few weeks


### Where the snapshots are located

The NIRD `$HOME` and NS project snapshots are available under:
- `/nird/home/u1/.snapshots`
- `/nird/projects/NSxxxxK/.snapshots`

The snapshots of the NN project backups from Betzy, Fram and Saga are available under:
- `/tos-project4/fram/backups/.snapshots`
- `/trd-project4/saga/backups/.snapshots`
- `/trd-project4/betzy/backups/.snapshots`


### How to recover deleted/overwritten data

A deleted/overwritten file in the home directory on NIRD can be recovered like this:
```console
$ cp /nird/home/u1/.snapshots/DATE/$USER/mydir/myfile /nird/home/$USER/mydir/
```

Note that snapshots are taken every night only. This means that deleted files
which did not exist yet yesterday cannot be recovered from snapshots.

To recover a deleted or overwritten file `/nird/projects/NSxxxxK/dataset1/myfile`,
you can copy a snapshot back to the folder and restore the deleted/overwritten file:
```console
$ cp /nird/projects/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/NSxxxxK/dataset1/
```

Select the DATE accordingly to your case.

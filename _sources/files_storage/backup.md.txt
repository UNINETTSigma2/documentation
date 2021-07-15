# Backup on Betzy, Fram, Saga and NIRD

```{warning}
**Only home directories with enforced quotas are backed up**

Any \$HOME area using more than 20 GiB or more than 100 000 files is
not backed up. To have your \$HOME backed up, you need to shrink the
disk usage below the 20 GiB limit *and* below 100 000 files *and*
[notify support](/getting_help/support_line.md).

If "dusage" reports 0 Bytes limits, this means that you **do not have disk quotas**
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

The work area also enforces an autocleanup strategy, and is **not** meant for permanent storage.
Files in this area will be **deleted** after 42 or 21 days, depending on the storage capacity,
see [User work area](user-work-area) for details.

### Where the backups are located

**HOME**: The nightly backups of the `$HOME` areas on Betzy, Fram and Saga end up in your `$HOME` area on
NIRD, which is geo-replicated between NIRD-TOS (Tromsø) and NIRD-TRD (Trondheim):
- `/nird/home/$USER/backups/betzy`
- `/nird/home/$USER/backups/fram`
- `/nird/home/$USER/backups/saga`

**PROJECT**: The nnXXXXk project areas on Betzy, Fram and Saga are backed up only to the
local NIRD site for each machine, and end up in the following folders on NIRD:
- `/tos-project4/fram/backups`
- `/trd-project4/saga/backups`
- `/trd-project4/betzy/backups`

```{note}
All users on either Betzy, Fram or Saga should automatically get access also to NIRD,
and can log in to either of the NIRD sites with

- `ssh <user>@login-tos.nird.sigma2.no` for NIRD-TOS
- `ssh <user>@login-trd.nird.sigma2.no` for NIRD-TRD
```

### How to recover deleted/overwritten data

**HOME**: To recover a deleted or overwritten file in `/cluster/home/$USER` on either Betzy, Fram or Saga
go to your home directory on NIRD, go to the backup folder and then browse in the directory
corresponding to the HPC system you come from for the file you want to restore.
You can then use `scp`/`sftp` to transfer a copy to your home directory on the HPC system.

**PROJECT**: The nightly backup from the Betzy, Fram and Saga project directories (`/cluster/projects/nnXXXXk`)
can be accessed *directly* from the respective machines, or from NIRD:

- `/tos-project4/fram/backups` from Fram or NIRD
- `/trd-project4/saga/backups` from Betzy, Saga or NIRD
- `/trd-project4/betzy/backups` from Betzy, Saga or NIRD

The Saga and Betzy backups can be accessed mutually from both machines, since both are located
on the Trondheim side of NIRD, but the Fram backup cannot be accessed from Saga or Betzy
(and vice versa).


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

```
$ cp /nird/home/u1/.snapshots/DATE/$USER/mydir/myfile /nird/home/$USER/mydir/
```

Note that snapshots are taken every night only. This means that deleted files
which did not exist yet yesterday cannot be recovered from snapshots.

To recover a deleted or overwritten file `/nird/projects/NSxxxxK/dataset1/myfile`,
you can copy a snapshot back to the folder and restore the deleted/overwritten file:

```
$ cp /nird/projects/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/NSxxxxK/dataset1/
```

Select the DATE accordingly to your case.



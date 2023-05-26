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

**HOME**: The nightly backups of the `$HOME` areas on Betzy, Fram and Saga end up in your `$HOME` area on NIRD :
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



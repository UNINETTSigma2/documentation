(storage-backup)=

# Backup on Betzy, Saga, and NIRD


## Backup of home folders on compute clusters

**Betzy and Saga**: Home folder is backed up to NIRD storage, and last copy of user home can be accessed via following mount point on corresponding clusters login nodes:
- `/cluster/backup/home/$username`

Please, note that we keep a copy of all files, but backup only copies the changes (done through a rsync process). If the file has not been modified, the timestamp will remain the same as on the file inside your home or project folders.

All other areas are not backed up. See the {ref}`storage areas section <storage-areas>` for reference.

In addition to not being backed up, the work area `/cluster/work` also enforces
an automatic cleanup strategy, and is **not** meant for permanent storage.
Files in this area will be **deleted** after 42 or 21 days, depending on the storage capacity,
see [User work area](user-work-area) for details.

## Snapshots

In addition to the daily backup, we also have snapshots of all home files, copied fully:

**Location**: `/cluster/backup/home/.snapshots/`
- Daily snapshots for the last 7 days
- Weekly snapshots for the last 6 weeks


## Backup on NIRD

Protection against data corruption on NIRD is implemented by taking nightly snapshots. Even so, it is the responsibility of the PI/XO to regulate the usage and take steps to ensure that the data are adequately secured against human errors or inappropriate usage/access.

The allocated storage quota on NIRD is meant for primary storage. Backup to a secondary location is a service on demand and can be ordered for selected datasets.

Snapshots and backup service on NIRD are described in details on the dedicated pages linked below.

```{eval-rst}
.. toctree::
   :maxdepth: 1

   nird/snapshots_lmd.md
   nird/backup_lmd.md
```

(storage-backup)=

# Backup on Betzy, Fram, Saga, and NIRD


## Backup of home folders on compute clusters

**Betzy, Fram and Saga**: Home folder is backed up daily to NIRD storage, and can be accessed via following mount point on corresponding clusters login nodes:
- `/cluster/backup/home/$username`
Please, note that only modified and new files are copied. You might see an earlier date on your files if you have not modified them.

## Backup of project folders on compute clusters 

Directories under `/cluster/projects` are backed up. All other areas are not backed up.

In addition to not being backed up, the work area `/cluster/work` also enforces
an automatic cleanup strategy, and is **not** meant for permanent storage.
Files in this area will be **deleted** after 42 or 21 days, depending on the storage capacity,
see [User work area](user-work-area) for details.

**Betzy,Fram and Saga**: The project areas are backed up to NIRD storage which can be accessed via following mount point on all clusters login nodes:
- `/cluster/backup/hpc/betzy/nnXXXXk`
- `/cluster/backup/hpc/fram/nnXXXXk`
- `/cluster/backup/hpc/saga/nnXXXXk`

## Snapshots

In addition to the daily backup, we also have snapshots of all project files copied fully:
**Location**: `/cluster/backup/hpc/.snapshots/`
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

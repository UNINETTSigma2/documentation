(storage-backup)=

# Backup on Betzy, Fram, Saga, and NIRD


## Backup of home folders on compute clusters

```{warning}
Since storage moved from old NIRD to new NIRD, **home folders are currently not
backed up**.
```

We are working on enabling cluster home backup to NIRD as soon as possible.
As soon as this is in place, we will update both the documentation and also
<https://opslog.sigma2.no/>.


## Backup of project folders on compute clusters 

Directories under `/cluster/projects` are backed up. All other areas are not backed up.

In addition to not being backed up, the work area `/cluster/work` also enforces
an automatic cleanup strategy, and is **not** meant for permanent storage.
Files in this area will be **deleted** after 42 or 21 days, depending on the storage capacity,
see [User work area](user-work-area) for details.

**Fram and Betzy**: The project areas are backed up to Saga:
- `/cluster/backup/betzy/projects/nnXXXXk`
- `/cluster/backup/fram/projects/nnXXXXk`

**Saga**: The project areas are backed up to NIRD.

If you have project access to the cluster where backups are stored, then you
can retrieve them yourself. If you cannot access the cluster that holds the
project backups, please contact support and we will help you restoring your
data.


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

# Backup

**To be implemented by production date.**

Both home directories (`/nird/home`) and project areas (`/nird/projects`) have
backup in the form of snapshots and geo-replication.

Snapshots are taken with the following frequencies:
* `/nird/home`: 
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
* `/nird/projects`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

Scratch areas (`/cluster/work` and `/node/scratch`) does *not* have any backup.

## List snapshots

Snapshots are available at the following places:
* `/nird/home`: 
  - /nird/home/u1/.snapshots
* `/nird/projects`:
  - /nird/projects/nird/NSxxxxK/.snapshots

## Restore data

Documentation to be added


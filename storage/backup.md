# Backup

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

To recover the file 

     /projects/NSxxxxK/dataset1/myfile

inadvertently deleted (or to recover an older version) do

    cp /nird/projects/nird/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/nird/NSxxxxK/dataset1/
    
select the date accordingly to your case.
This procedure is applicable to files in home directories.      
Note that snapshots are taken every night only. So file created the day of the deletion   
cannot be recovered with snapshots.


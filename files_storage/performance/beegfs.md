

# Storage performance: BeeGFS filesystem (Saga)

Striping in BeeGFS (`/cluster`) cannot be re-configured on Saga by users, it can currently
only be modified by system administrators.


## How to find out the current striping

To check current stripe size, use:
```
$ beegfs-ctl --getentryinfo [file_system, dir, file]
```

For example to check your home folder stripe size on Saga, you can do:
```
$ beegfs-ctl --getentryinfo /cluster/home/$HOME
```

For example to check file tripe size:
```
$ beegfs-ctl --getentryinfo /cluster/tmp/test

EntryID: 5-5DC49168-19C
Metadata node: mds4-p1-m2 [ID: 412]
Stripe pattern details:
+ Type: RAID0
+ Chunksize: 512K
+ Number of storage targets: desired: 4; actual: 4
+ Storage targets:
  + 4201 @ oss-4-4-stor2 [ID: 42]
  + 4202 @ oss-4-4-stor2 [ID: 42]
  + 4203 @ oss-4-4-stor2 [ID: 42]
  + 1101 @ oss-4-1-stor1 [ID: 11]
```

This shows that this particular file is striped over 4 object storage targets.

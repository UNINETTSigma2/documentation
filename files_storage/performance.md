(storage-performance)=

# Optimizing storage performance


## What to avoid

- Avoid having a **large number of files in a single directory** and
  rather split files in multiple sub-directories.
- **Avoid repetitive `stat`** operations because it can create a significant
  load on the file system.
- **Do not use `ls -l`** on large directories, because it can be slow.  Rather
  use `ls` and run `ls -l` only for the specific files you need
  extended information about.


## Lustre file system (Betzy and Fram)

To get best throughput on the scratch file system (`/cluster/work`), you may
need to change the data striping. Striping shall be adjusted based on the
client access pattern to optimally load the object storage targets (OSTs).
On Lustre, the OSTs are referring to disks or storage volumes constructing the
whole file system.

The `stripe_count` indicates how many OSTs to use.
The `stripe_size` indicates how much data to write to one OST before moving to
the next OST.

* Striping will only take affect *only* on new files, created or copied
  into the specified directory or file name.
* Default `stripe_count` on `/cluster` file system on Fram is 1.
* Betzy is implementing Progressive File Layouts to dynamically set file stripe
  size based on file size growth.

For more detailed information on striping, please consult the
[Lustre](https://www.lustre.org/) documentation.

```{note}
**Betzy: Progressive File Layouts**

PFL removes the need to explicitly specify striping for each file, 
assigning different Lustre striping characteristics to contiguous 
segments of a ﬁle as it grows.
Dynamic striping allows lower overhead for small files and assures 
increased bandwidth for larger files.
However, note that for workloads with signiﬁcant random read phases it is
best to manually assign stripe size and count.

**Betzy: Data on Metadata**

Lustre file system performance is optimized for large files. To balance
that, data on metadata (DoM) is enabled on Betzy to ensure higher
performance in case of frequently accessed small files.
Files accessed with a size of 2KB or smaller will be stored on a very
fast NVMe JBOD directly connected to the metadata servers.
```


### How to find out the current striping

To see the current stripe size (in bytes), use `lfs getsripe [file_system, dir, file]`
command. e.g.:
```console
$ lfs getstripe example.txt

example.txt
lmm_stripe_count:  1
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    0
lmm_stripe_offset: 75
	obdidx		 objid		 objid		 group
	    75	      54697336	    0x3429d78	             0
```


### Rules of thumb to set stripe counts

For best performance we urge you to always profile the I/O characteristics of
your HPC application and tune the I/O behavior.

Here is a list of rules you may apply to set stripe count for
your files:
- files smaller than 1 GB: default striping
- files size between 1 GB - 10 GB: stripe count 2
- files size between 10 GB - 1 TB: stripe count 4
- files bigger than 1 TB: stripe count 8


### Large files

For large files it is advisable to increase stripe count and perhaps chunk size,
too. e.g.:
```bash
# stripe huge file across 8 OSTs
$ lfs setstripe --stripe-count 8 "my_file"

# stripe across 4 OSTs using 8 MB chunks.
$ lfs setstripe --stripe-size 8M --stripe-count 4 "my_dir"
```

It is advisable to use higher stripe count for applications that
write to a single file from hundreds of nodes, or a binary executable that
is loaded by many nodes when an application starts.

Choose a stripe size between 1 MB and 4 MB for sequential I/O. Larger than 4 MB
stripe size may result in performance loss in case of shared files.

Set the stripe size a multiple of the write() size, if your application is
writing in a consistent and aligned way.


### Small files

For many small files and one client accessing each file, change stripe count to 1.
Avoid having small files with large stripe counts. This negatively impacts the
performance due to the unnecessary communication to multiple OSTs.
```console
$ lfs setstripe --stripe-count 1 "my_dir"
```


## BeeGFS filesystem (Saga)

Striping in BeeGFS (`/cluster`) cannot be re-configured on Saga by users, it can currently
only be modified by system administrators.

But one thing you can do is to check the current stripe size (here for an example file):
```console
$ beegfs-ctl --getentryinfo example.txt

Entry type: file
EntryID: 55-628B3E1D-144
Metadata node: mds3-p2-m4 [ID: 324]
Stripe pattern details:
+ Type: RAID0
+ Chunksize: 512K
+ Number of storage targets: desired: 4; actual: 4
+ Storage targets:
  + 1203 @ oss-4-1-stor2 [ID: 12]
  + 2101 @ oss-4-2-stor1 [ID: 21]
  + 2102 @ oss-4-2-stor1 [ID: 21]
  + 2103 @ oss-4-2-stor1 [ID: 21]
```

This shows that this particular file is striped over 4 object storage targets.

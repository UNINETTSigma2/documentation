# Performance tuning tips
##Luster Filesystem 
To get best throughput on the scratch file system (/cluster/work), you may
need to change the data striping. Striping shall be adjusted based on the
client access pattern to optimally load the object storage targets (OSTs).
On Lustre, the OSTs are referring to disks or storage volumes constructing the
whole file system.

The `stripe_count` indicates how many OSTs to use.
The `stripe_size` indicates how much data to write to one OST before moving to 
the next OST.

**Note**: 
* striping will only take affect *only* on new files, created or copied
 into the specified directory or file name;
* default stripe_count on /cluster is 1.

For more detailed information on striping, please consult the
[Lustre](http://lustre.org) documentation.

### Check out current striping

To see current stripe size, use `lfs getsripe [file_system, dir, file]`
command. e.g.:

```
lfs getstripe /cluster/tmp/test
/cluster/tmp/test
stripe_count:   1 stripe_size:    1048576 stripe_offset:  -1
```
**Note**: stripe size is shown in bytes.

### Large files

For large files it is advisable to increase stripe count and perhaps chunk size
too. e.g.:

```
# stripe huge file across 8 OSTs
lfs setstripe --stripe-count 8 "my_file"

# stripe across 4 OSTs using 8MB chunks.
lfs setstripe --stripe-size 8M --stripe-count 4 "my_dir"
```

It is advisable to use higher stripe count for scientific application that 
writes to a single file from hundreds of nodes, or a binary executable that 
is loaded by many nodes when an application starts.

Choose a stripe size between 1MB and 4MB for sequential I/O. Larger than 4MB 
stripe size may result in performance loss in case of shared files.

Set the stripe size a multiple of the write() size, if your application is
writing in a consistent and aligned way.

### Small files

For many small files and one client accessing each file, change stripe count to 1.
Avoid having small files with large stripe counts. This negatively impacts the 
performance due to the unnecessary communication to multiple OSTs.

    lfs setstripe --stripe-count 1 "my_dir"
## BeeGFS filesystem 
Striping in BeeGFS can be configured on a per-directory and per-file basis. To check current stripe szie, use 

`beegfs-ctl --getentryinfo [file_system, dir, file]`
 
For example to check your home folder stripe size on Saga, you can do:

```beegfs-ctl --getentryinfo /cluster/home/$HOME```
## DON'Ts

* Avoid having a large number of files in a single directory and rather split
	files in multiple sub-directories.
* Avoid accessing small files. Use `/node/scratch` whenever possible instead
	of `/cluster/work` for small files.
* Avoid repetitive `stat` operations because it creates a significant load on
	the file system.
* Do not use `ls -l` on whole directory because it issues RPCs to the MDS for
	each file and directory it lists. Use rather `ls` and run `ls -l` only for
	the specific file you need extended information such as: permissions,
	ownership, etc.

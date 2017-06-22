# Performance tuning tips

To get best throughput on the scratch file system (/cluster/work), you may
need to change the data striping. Striping shall be adjusted based on the
client access pattern to optimally load the object storage targets (OSTs).
On Lustre, the OSTs are referring to disks or storage volumes constructing the
whole file system.

**Note**: striping will only take affect *only* on new files, created or copied into the specified directory or file name.

For more detailed information on striping, please consult the
[Lustre](http://lustre.org) documentation.

## Large files

For large files it is advisable to increase stripe count and perhaps chunk size
too. e.g:

```
# stripe huge file across 8 OSTs
lfs setstripe --count 8 "my_file"
# stripe across 4 OSTs using 8MB chunks.
lfs setstripe --size 8M --count 4 "my_dir" 
```

## Small files

For many small files and one client accessing each file, change stripe count to 1.

    lfs setstripe --count 1 "my_dir"



# Storage performance: Lustre file system

## Betzy, Fram and Stallo

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
* Default `stripe_count` on `/cluster` file system on Fram and Stallo is 1.
* Betzy is implementing Progressive File Layouts to dynamically set file stripe
	size based on file size growth.

<div class="alert alert-info">
  <h4>Betzy: Progressive File Layouts</h4>
  <p>
	    PFL removes the need to explicitly specify striping for each file, 
			assigning different Lustre striping characteristics to contiguous 
			segments of a ﬁle as it grows.
			Dynamic striping allows lower overhead for small files and assures 
			increased bandwidth for larger files.
			However, note that for workloads with signiﬁcant random read phases it is
			best to manually assign stripe size and count.
	</p>
</div>

* Betzy implements another new feature, called data on metadata for small files
	sith size under 2KB.

<div class="alert alert-info">
  <h4>Betzy: Data on Metadata</h4>
	<p>
		 Lustre file system performance is optimized for large files. To balance
		 that, data on metadata (DoM) is enabled on Betzy to ensure higher
		 performance in case of frequently accessed small files.
		 Files accessed with a size of 2KB or smaller will be stored on a very
		 fast NVMe JBOD directly connected to the metadata servers.
  </p>
</div>

For more detailed information on striping, please consult the
[Lustre](http://lustre.org) documentation.


### How to find out the current striping

To see current stripe size (in bytes), use `lfs getsripe [file_system, dir, file]`
command. e.g.:

```
$ lfs getstripe /cluster/tmp/test

/cluster/tmp/test
stripe_count:   1 stripe_size:    1048576 stripe_offset:  -1
```


### Large files

For large files it is advisable to increase stripe count and perhaps chunk size
too. e.g.:

```bash
# stripe huge file across 8 OSTs
$ lfs setstripe --stripe-count 8 "my_file"

# stripe across 4 OSTs using 8 MB chunks.
$ lfs setstripe --stripe-size 8M --stripe-count 4 "my_dir"
```

It is advisable to use higher stripe count for scientific application that
writes to a single file from hundreds of nodes, or a binary executable that
is loaded by many nodes when an application starts.

Choose a stripe size between 1 MB and 4 MB for sequential I/O. Larger than 4MB
stripe size may result in performance loss in case of shared files.

Set the stripe size a multiple of the write() size, if your application is
writing in a consistent and aligned way.


### Small files

For many small files and one client accessing each file, change stripe count to 1.
Avoid having small files with large stripe counts. This negatively impacts the
performance due to the unnecessary communication to multiple OSTs.

```bash
$ lfs setstripe --stripe-count 1 "my_dir"
```

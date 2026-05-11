
(data-transfer)=

# Data transfer

This page covers all common ways to move data to, from, and between Sigma2 systems, covering both POSIX-based file transfer (rsync, rclone, scp) and object storage via the S3 protocol (NIRD S3, AWS, etc.).

**Jump to**: {ref}`posix-transfer` | {ref}`s3-transfer`

```{admonition} Summary: choosing the right tool

| Dataset size | Scenario | Recommended tool |
|---|---|---|
| Small |Laptop to HPC cluster (few files) | `rsync` |
| Small | Laptop to HPC cluster (many small files) | `rclone` or `rsync` |
| Small | Between HPC clusters / NIRD (POSIX) | `cp` / `mv` (on login node), or `rsync` |
| Large | Between HPC clusters / NIRD (POSIX) | `rclone` with `--transfers` |
| Small | NIRD to HPC cluster (S3) | `aws cli` or `rclone` |
| Small | Public Cloud to HPC cluster (S3) | `aws cli` or `rclone` |
| Large | NIRD to HPC cluster (S3) | `s5cmd` |
| Any | Programmatic access | `boto3` (Python) |

```

Data can be moved to, from, and between Sigma2 systems using two broad approaches:

- **POSIX file transfer** (rsync, rclone, scp) — transfers data over SSH using the regular file system.
- **S3 object storage transfer** (aws CLI, rclone, s5cmd, boto3) — transfers data using the S3 protocol, available for NIRD Data Lake and external cloud storage such as AWS S3.

---
(posix-transfer)=
## POSIX file transfer

```{admonition} Summary: use rsync for file transfer

For file transfer to/from and between compute and storage systems (Betzy, Fram,
Saga, NIRD), **we recommend `rsync`**. This tool is often faster than `scp` (for
many small files and it does not copy files that are already there) and
potentially also safer against accidental file overwrites.
For more details, see {ref}`advantages-over-scp`.

When using `rsync`, there is **no need to zip/tar files first**.

On Windows, many other tools exist ([WinSCP](https://winscp.net/),
[FileZilla](https://filezilla-project.org/),
[MobaXterm](https://mobaxterm.mobatek.net/), and others), but we recommend to
use `rsync` through [Windows Subsystem for Linux
(WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

For large data transfers `rclone` is an option which offers better performance through
multiple parallel transfers, see {ref}`rclone-posix`.

For transfers to and from NIRD, the S3 protocol is also an option. It can reach
very high performance as all object transfers are independent of each other, see
{ref}`s3-transfer`.
```

**NB**: Since the implementation of 2FA (2 factor authentication) on Sigma2 clusters,
you might experience issues while using Filezilla, in which you never get asked to
provide the 2fa key before the password. To fix this, (in Filezilla) you need to go
to `Site Manager` and change `Protocol Type` to `Interactive`.


### Transferring files between your computer and a compute cluster or storage resource

This is a good starting point but below we will explain what these components
and options mean:

```console
$ rsync --info=progress2 -a file-name       username@cluster:receiving-directory
$ rsync --info=progress2 -a directory-name/ username@cluster:receiving-directory/directory-name
```

- `--info=progress2`: This will show progress (how many percent, how much time
  left). You can also leave it out if you don't need to know how far the
  copying is. There is also a `--progress` option but that one will show
  progress for each file individually and often you rather want to know the
  overall progress.
- `-a`: Preserves ownership and time stamp and includes the `-r` option which copies
  folders recursively.
- `file-name` or `directory-name`: These are on your computer and you want to
  transfer them to the receiving server.
- `username`: Your username on the remote cluster. If your usernames on your
  local computer and on the remote resource are the same, you can leave out the
  `username@` part.
- `cluster`: The remote server. For example: `saga.sigma2.no`.
- `receiving-directory`: The directory on the remote server which will receive the file(s) and/or directories.

If you want to make sure that `rsync` does not overwrite files that are newer
on the receiving end, add the `--update` option.

If you want to `rsync` between two computers that both offer an SSH connection, note that then
you can use `rsync` both ways: from cluster A to cluster B, but also the reverse.

````{admonition} rsync directory
Please note that there is a trailing slash (`/`) at the end of the first argument in the
syntax of the second command, while rsync directories, ie:

```console
$ rsync --info=progress2 -a directory-name/ username@cluster:receiving-directory/directory-name
```
This trailing slash (`/`) signifies the contents of the directory `directory-name`.
The outcome would create a hierarchy like the following on your cluster:
```console
~/receiving-directory/directory-name/contents-of-the-dir
```

Without the trailing slash, `directory-name`, including the directory, would be placed within your receiving directory.
The outcome would be the following on the cluster:
```console
~/receiving-directory/directory-name/directory-name/contents-of-the-dir
```
````


### rsync using compression

If you have a strong CPU at both ends of the line, and you're on a slow
network, you can save bandwidth by compressing the data with the `-z` flag:

```console
$ rsync --info=progress2 -az file-name      username@cluster:receiving-directory
$ rsync --info=progress2 -az directory-name username@cluster:receiving-directory/directory-name
```


### Problem with many small files

Many small files are often not great for the transfer (although `rsync` does
not seem to mind but for `scp` this can make a big difference, see below). Many
tiny files are often also a problem for parallel file systems. If you develop
programs for high-performance computing, avoid using very many tiny files.


(advantages-over-scp)=

### Advantages over scp and similar tools

- `rsync` will not transfer files if they already exist and do not differ.
- With `rsync --update` you can avoid accidentally overwriting newer files in the destination directory.
- You can use compression for file transfer.
- Resumes interrupted transfers.
- More flexibility and better cross-platform support.

Typically people recommend `scp` for file transfer and we have also done this
in the past. But let us here compare `scp` with `rsync`. In this example I
tried to transfer a 100 MB file from my home computer (not on the fast
university network) to a cluster, either as one large file or split into 5000
smaller files.

For one or few files it does not matter:

```bash
$ scp file.txt username@cluster:directory
# 81 sec

$ rsync --info=progress2 -a file.txt username@cluster:directory
# 79 sec

$ rsync --info=progress2 -az file.txt username@cluster:directory
# 61 sec
```

However, **it can matter a lot if you want to transfer many small files**.
Notice how the transfer takes 10 times longer with `scp`:

```{code-block} bash
---
emphasize-lines: 2, 5
---
$ scp -r many-files username@cluster:directory
# 833 sec

$ rsync --info=progress2 -a many-files username@cluster:directory/many-files
# 81 sec

$ rsync --info=progress2 -az many-files username@cluster:directory/many-files
# 62 sec
```

In the above example, `scp` struggles with many small files but `rsync` does
not seem to mind. For `scp` we would have to first `tar`/`zip` the small files
to one large file but for `rsync` we don't have to.

````{admonition} How was the test data created?
Just in case anybody wants to try the above example on their own, we used this
script to generate the example data:

```bash
#!/usr/bin/env bash

# create a file that is 100 MB large
base64 /dev/urandom | head -c 100000000 > file.txt

# split into 5000 smaller files
mkdir -p many-files
cd many-files
split -n 5000 ../file.txt
```
````


### Transferring files between Betzy/Olivia/Saga and NIRD

The easiest way to transfer files between clusters is to set up an
[ssh-keypair](https://documentation.sigma2.no/getting_started/ssh.html#connecting-to-a-server).
Once (following the guide) there's a private/public key on sender/receiver,
you won't need to authenticate with password + OTP. Note that the guide above
is currently only limited to cross-cluster connections/transfers.

Since NIRD is mounted on the login nodes of Betzy, Olivia, and Saga,
one can use regular `cp` or `mv` commands on the cluster login nodes to copy or
move files into or out of the NIRD project areas.

| System | Mount point |
|---|---|
| NIRD Data Peak | `/nird/datapeak/NSxxxxK` |
| NIRD Data Lake | `/nird/datalake/NSxxxxK` |

```{note}
On Saga and Betzy, NIRD is mounted on **login nodes only** (not compute nodes).
On Olivia, it is mounted on SVC nodes (read-write) and compute nodes (read-only).
Olivia also supports automatic data staging via Slurm --
see [Staging In/Out Files from/to NIRD using Slurm](https://documentation.sigma2.no/files_storage/clusters.html).
```

For more information, please check out the page about {ref}`storage-areas`.


### What to do if rsync is not fast enough?

Disk speed, meta-data performance, network speed, and firewall speed may limit
the transfer bandwidth.

If you have access to a network with a large bandwidth and you are sure that
you are limited by the one `rsync` process and not by something else, you can
start multiple `rsync` processes, by piping a list of paths to `xargs` or
`parallel` which launches multiple `rsync` instances in parallel. But please
mind that this way you can saturate the network bandwidth for other users and
also saturate the login node with `rsync` processes or overwhelm the file
system. If you have to transfer large amount of data and one `rsync` process is
not enough, we recommend that you talk to us first: {ref}`support-line`.

Please also **plan for it**: If you need to transfer large amount of data,
don't start on the last day of your project. Data transfer may take hours or
even days.


(rclone-posix)=

### *rclone* as a faster alternative

While rsync does a good job, it unfortunately only uses one thread (or transfer),
while *rclone* can use a range of parallel transfers (both one per file and split
a large file into chunks).

`rclone` can utilise multiple threads / streams to run multiple transfers in
parallel. An example copying the same 5000-file dataset used above (this does
the transfer from Saga to Olivia):

```console
$ rclone copy SAGA:/cluster/work/users/user/many-files . -P --transfers=30 --ignore-checksum
```

The following table is illustrative of the performance:

| Transfer application | Options / threads | Wall time [seconds] |
|---|---|---|
| scp | | 833 |
| rsync | `-a` | 81 |
| rsync | `-az` | 62 |
| rclone | `--transfers=10` | 25 |
| rclone | `--transfers=20` | 15 |
| rclone | `--transfers=30` | 11 |
| rclone | `--transfers=40` | 10 |
| rclone | `--transfers=50` | 9 |

`rclone` can keep a large number of operations in flight simultaneously which
is the reason for it being so efficient.

A command like:

```console
$ rclone copy SAGA:/cluster/projects/nnXXXXk/user/ . -P --transfers=20
```

will copy 20 files in parallel. A larger example:

```console
$ rclone copy SAGA:/cluster/projects/nnxxxxk/ . -P --transfers=60 --ignore-checksum
Transferred:   200 GiB / 200 GiB, 100%, 2.038 GiB/s, ETA 0s
Checks:          0 / 0, -,  Listed 200
Transferred:   200 / 200, 100%
Elapsed time:  1m33.7s
```

This shows close to 2 GBytes/s -- about 7 TBytes per hour, or 150 TBytes/day.


### Troubleshooting: "Broken pipe" error during transfer

The organization which provides the network to the clusters, may perform daily
housekeeping of their [DNS](https://en.wikipedia.org/wiki/Domain_Name_System)
and then the connection from outside to the Sigma2 services can drop. This can
cause a "broken pipe" error during file transfer from outside.

One way to avoid this, especially while copying large datasets, is to use IP
addresses instead of domain names.

One way to get the IP of one of the login nodes (example: Saga):

```console
$ nslookup saga.sigma2.no
```

---

(s3-transfer)=

## S3 object storage transfer

The S3 protocol is available for NIRD Data Lake  {ref}`nird-s3` 
(including the Central Data Library {ref}`nird-cdl`)
and for external cloud storage such as AWS S3. S3 transfers are independent per
object, enabling very high parallel throughput -- NIRD S3 is capable of up to 27 GB/s.

S3 is the right choice when:

- Your data lives in NIRD Data Lake or another S3-compatible store.
- You need to fetch data from AWS or another cloud provider.
- You are building automated or scripted pipelines.
- You are integrating with AI/ML frameworks that expect object storage.


### Proxy setup (required on all Sigma2 systems)

All outbound HTTPS traffic from Sigma2 HPC systems goes through a proxy. **Set
these variables before using any S3 tool**, or add them to your `~/.bashrc`.

**Olivia:**

```bash
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
```

**Saga :**

```bash
export http_proxy=http://proxy.saga:3128/
export https_proxy=http://proxy.saga:3128/
```

Verify connectivity before starting a transfer:

```console
$ curl -I https://s3.nird.sigma2.no
```

A successful response returns `HTTP/1.1 200 OK`.

```{note}
This already works on Betzy without additional configuration.
```

### Credentials

**NIRD S3**

Your S3 credentials are provided in a file `<username>-<project>-s3creds.txt`
in your home directory after S3 access is activated. Configure `~/.aws/credentials`:

```ini
[default]
aws_access_key_id     = <your-access-key>
aws_secret_access_key = <your-secret-key>

[s3test]
aws_access_key_id     = <Access Key from the credentials file>
aws_secret_access_key = <Secret Key from the credentials file>
```

And `~/.aws/config`:

```ini
[default]
region = us-east-1

[profile s3test]
region = us-east-1
s3 =
  multipart_chunksize        = 5GB
  multipart_threshold        = 2GB
  max_concurrent_requests    = 100
```

**AWS S3**

```console
$ aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region (e.g. `eu-west-1`),
and output format. Or export credentials directly:

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```


### Using the AWS CLI

The AWS CLI works for both NIRD S3 (using `--endpoint-url`) and standard AWS S3.

```console
$ module load awscli
```

**NIRD S3:**

```bash
# List your buckets
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no s3 ls

# List objects in a bucket
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3 ls s3://username-nsxxxxk-bucketname/

# Download a single file
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3 cp s3://username-nsxxxxk-bucketname/path/to/file.nc ./file.nc

# Download a directory recursively
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3 cp s3://username-nsxxxxk-bucketname/dataset/ ./dataset/ --recursive

# Sync a prefix to local storage
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3 sync s3://username-nsxxxxk-bucketname/input/ ./input/

# Fetch object metadata
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3api head-object --bucket username-nsxxxxk-bucketname --key somefile

# Upload a file
aws --profile s3test --endpoint-url https://s3.nird.sigma2.no \
  s3 cp local-file.nc s3://username-nsxxxxk-bucketname/
```

**AWS S3** (no `--endpoint-url` needed):

```bash
# List buckets
aws s3 ls

# Download a single file
aws s3 cp s3://your-bucket-name/path/to/file.nc ./file.nc

# Download a directory recursively
aws s3 cp s3://your-bucket-name/dataset/ ./dataset/ --recursive

# Sync a bucket prefix to local
aws s3 sync s3://your-bucket-name/path/ ./local-path/

# Use a named profile
aws s3 ls --profile my-project-profile
```


### *rclone* with S3 backend

`rclone` also connects to S3 endpoints and offers the same parallel transfer
advantage as in the POSIX case. Measured throughput on NIRD S3 reaches ~10 GB/s
with sufficient parallelism.

Add the following to `~/.config/rclone/rclone.conf`:

```ini
[S3]
type                = s3
provider            = Ceph
env_auth            = false
access_key_id       = <your-access-key>
secret_access_key   = <your-secret-key>
endpoint            = https://s3.nird.sigma2.no
```

```bash
# List objects
rclone ls S3:<user>-ns<project>k-<user>/

# Download a single file (with progress)
rclone copy S3:user-nsXXXXk-user/tmp.medium . -P

# Download multiple files in parallel (wildcards don't work -- use --include)
rclone copy S3:user-nsXXXXk-user/ . --include "*.nc" -P --transfers=20
```

Example output:

```
Transferred:   245.027 GiB / 245.027 GiB, 100%, 19.587 MiB/s, ETA 0s
Transferred:   1 / 1, 100%
Elapsed time:  4m56.9s
```


### *s5cmd* -- maximum throughput for many objects

[`s5cmd`](https://github.com/peak/s5cmd) is significantly faster than the AWS CLI
for large numbers of files or objects, running operations fully in parallel.

```bash
# Install if not available as a module
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz

# Download from NIRD S3
./s5cmd --endpoint-url https://s3.nird.sigma2.no \
  cp 's3://username-nsxxxxk-bucketname/path/*' ./local-path/

# Download from AWS S3
./s5cmd cp 's3://your-bucket-name/path/*' ./local-path/
```


### Python (boto3)

For scripted or automated workflows, use `boto3`:

```bash
pip install boto3 --user
```

**NIRD S3:**

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.nird.sigma2.no",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
)

# List objects
response = s3.list_objects_v2(Bucket="username-nsxxxxk-bucketname")
for obj in response.get("Contents", []):
    print(obj["Key"])

# Download a file
s3.download_file("username-nsxxxxk-bucketname", "path/to/file.nc", "local-file.nc")
```

**AWS S3:**

```python
import boto3

s3 = boto3.client("s3", region_name="eu-west-1")
s3.download_file("your-bucket-name", "path/to/file.nc", "local-file.nc")
```

```{note}
When running Python scripts on HPC, make sure the proxy environment variables
are set in your job script or shell session.
```


### Example Slurm job script

The following fetches input data from NIRD S3, runs an analysis, and pushes
results back -- all within a single Slurm job on Saga:

```bash
#!/bin/bash
#SBATCH --job-name=fetch-and-run
#SBATCH --account=nnXXXXk
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1

# Proxy (required on Saga)
export http_proxy=http://proxy.saga:3128/
export https_proxy=http://proxy.saga:3128/

module load awscli

export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Fetch input data
aws --endpoint-url https://s3.nird.sigma2.no s3 sync \
  s3://username-nsxxxxk-bucketname/input-data/ $SCRATCH/input-data/

# Run analysis
python my_analysis.py --input $SCRATCH/input-data/

# Push results back
aws --endpoint-url https://s3.nird.sigma2.no s3 sync \
  $SCRATCH/results/ s3://username-nsxxxxk-bucketname/results/
```


### Troubleshooting S3 transfers

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not connect to endpoint URL` | Proxy not set | Export `http_proxy` and `https_proxy` |
| `SSL certificate verify failed` | Proxy TLS interception | Try `aws --no-verify-ssl` (use with caution) |
| `403 Forbidden` | Wrong credentials or no bucket permission | Check access key and bucket permissions |
| `NoSuchBucket` | Bucket name typo | Run `aws s3 ls` to list available buckets |
| Slow transfer speed | Single-threaded tool | Switch to `rclone --transfers=N` or `s5cmd` |
| Transfer interrupted | Network drop / DNS housekeeping | Re-run; use `rsync` instead of `cp` to resume |
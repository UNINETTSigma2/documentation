(access to old NIRD)=

# Access to old NIRD

The old NIRD login nodes can be accessed via

```console
login-tos.nird.sigma2.no
```

```{warning}
Notice, that old NIRD can be accessed until the end of October 2023.
```

Migration of the data on $HOME shall be completed by the end of May 2023.
 Beyond this term the integrity of the data will be not guaranteed.
 If you need some advice or have some questions, please do not hesitate 
to contact us via support@nris.no.

## How do I copy/migrate my home from old NIRD to new NIRD?

You should take the opportunity to clean up your home directory and archive or
copy only data you actually want to keep.

One way to copy your entire home directory would be by using the `rsync` tool.
Log into old NIRD and

```console
$ rsync -avzh dirname username@login.nird.sigma2.no:~/old_home/
```
This will recursively copy your directory/file `dirname` into the directory `old_home` on new 
NIRD `$HOME` folder.
Please note that you need to provide your `username`. Also you need to provide your password
unless you have set up passwordless login.

You can read `rsync` options with `man`

```console
$ man rsync
```

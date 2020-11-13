# Changing file ownership

```{note}
**This is cluster specific**

This page is only relevant for Fram and Saga - **not Betzy**, as disk quotas on Betzy are based on directories instead of groups.
```

Since file permissions are persistent across the file system, it might be necessary to manually change the ownership of one or more files.
This page will show an example of how to change ownership on a file that was moved from `$HOME` to `$USERWORK` in order to update the disk quotas.

We have a file in our `$HOME` called "myfile.txt" which is 1 GiB in size that we're moving to `$USERWORK` for use in a job:

```
[username@login-3.SAGA ~/example]$ ls -l
total 1048576
-rw-rw-r-- 1 username username_g 1073741824 Nov 13 13:11 myfile.txt

[username@login-3.SAGA ~/example]$ mv myfile.txt /cluster/work/users/username

```

By checking our disk usage with `dusage` we can see that the file is still counted towards the `$HOME` quota:

```
[username@login-3.SAGA ~/example]$ dusage
No options specified, assuming "-a". For help, use "-h" option.

===============================================================================
Block quota usage on: SAGA
===============================================================================
File system    User/Group          Usage          Soft Limit     Hard Limit     
-------------------------------------------------------------------------------
username_g     $HOME               1.0 GiB        20.0 GiB       20.0 GiB       
username       username (u)        1.0 GiB        0 Bytes        0 Bytes        
username       username (g)        0 Bytes        0 Bytes        0 Bytes      
dgi            dgi (g)             0 Bytes        0 Bytes        0 Bytes        
===============================================================================
```

The reason for this is that the file is still owned by the `username_g` group, which is used for the `$HOME` quota.

```
[username@login-3.SAGA /cluster/work/users/username]$ ls -l
total 1048576
-rw-rw-r-- 1 username username_g 1073741824 Nov 13 13:11 myfile.txt
```

Files in `$USERWORK` should be owned by the default user group, in this - the group named `username`.
To change the file group ownership we can use the command `chgrp`:

```
[username@login-3.SAGA /cluster/work/users/username]$ chgrp username myfile.txt 
[username@login-3.SAGA /cluster/work/users/username]$ ls -l
total 1048576
-rw-rw-r-- 1 username username 1073741824 Nov 13 13:11 myfile.txt
```

The file is now owned by the correct group and we can verify that the disk quotas have been updated by running `dusage` again:

```
[username@login-3.SAGA /cluster/work/users/username]$ dusage
No options specified, assuming "-a". For help, use "-h" option.

===============================================================================
Block quota usage on: SAGA
===============================================================================
File system    User/Group          Usage          Soft Limit     Hard Limit     
-------------------------------------------------------------------------------
username_g     $HOME               0 Bytes        20.0 GiB       20.0 GiB       
username       username (u)        1.0 GiB        0 Bytes        0 Bytes        
username       username (g)        1.0 GiB        0 Bytes        0 Bytes      
dgi            dgi (g)             0 Bytes        0 Bytes        0 Bytes         
===============================================================================
```
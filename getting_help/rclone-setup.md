
# Use of `rclone` without using 2FA/OTP

## Install rclone (Linux)

```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

For precompiled binary, MacOS or Windows installation, please see https://rclone.org/install/


## Generate an ssh key pair and copy the pub key to the remote host

Run the command

```
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_rclone
```

Do not set any paraphrase when asked!!!

Once the key pair has been generated, log into the remote host (a cluster login node, e.g. _login.saga.sigma2.no_) and go to the directory `.ssh` in your `home` directory. Open the file `authorized_keys` and attach below the content of the file `id_ed25519_rclone.pub` - your public key which was generated along with your private key in your local machine. Be careful not to paste the content of your private key instead! If the file `authorized_keys` is missing, just create it.


## Configure `rclone`

Method A:

Create a directory/file in ~/.config

```
~/.config/rclone/rclone.config
```

with the following content:

``` 
# Any string which will represent the remote location, the example here is "saga" for Saga cluster, could be "olivia" etc
[saga]
type = sftp
# The host you will copy files to/from
host = login.saga.sigma2.no 
port = 12
# Absolute path to your private key file
key_file = /home/<USERNAME>/.ssh/id_ed25519_rclone
shell_type = unix
md5sum_command = none
sha1sum_command = none
known_hosts_file = none
```

Port 12 is the only one allowing for sftp connections without 2F authentication.

Method B: 

If you know `rclone` and prefer to configure it yourself, run 

```
rclone config
```

and follow the options by selecting the ones from the _rclone.config_ file here above. Select `default` for all the options you are not sure about.

## Running `rclone`

1. To copy directories

```
rclone copy -v </path/to/you/source/files/> remote:</path/to/you/target/files/>

e.g.
rclone copy -v /home/<username>/some-dir/ saga:/home/<username>/some-dir/>
```

where `saga` is the value for remote from your `rclone.config` file (top line) 

2. To copy a single file

Use `copyto` command

```
rclone copyto src dst
```

For an exhaustive list of commands, see https://rclone.org/#features




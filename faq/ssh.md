# SSH

Some SSH related frequently asked questions are documented down below.
For more in-depth details, other options, please consult the man pages: 
`man ssh` and `man ssh_config`.

## Password-Less Login

 To login to a server without typing in your password every time,
 you can configure ssh to use public key cryptography.
In case you use a linux system start by generating a pair of keys and
saving them in the folder `.ssh`:
```
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_cluster
```
You should provide a passphrase to encrypt the key.

To copy and install the public key to the server, for example saga,
we use:
```
ssh-copy-id -i ~/.ssh/id_cluster username@saga.sigma2.no
```

The passphrase secures the key but can be annoying if it has to be entered every time you use the key. `ssh-agent` is program installed on virtually all linux versions to manage the keys so that you only have to unlock the key once.
We can add the new key with:
```
ssh-add ~/.ssh/id_cluster
```

For more information see https://www.ssh.com/ssh/keygen/

## X11 forwarding

X11 forwarding should be used with caution due to security implications. Please
note that if someone can read your X authorization database [^1], that person 
would be able to access the local X11 display through the forwarded connection.

We suggest switching it on *only* when needed, with the use of options (`-X` or 
`-Y`)  passed to the `ssh` command. Whenever possible, use `-X` option to mark
remote X11 clients untrusted.

In some cases `-X` will fail to work and either the use of `-Y` option or 
setting `ForwardX11Trusted` in your ssh config file to "yes" is required. In 
this case remote X11 clients will have full access to the original X11 display.

Alternatively, if X11 forwarding is always needed, you can configure it on a 
per-host basis in your `.ssh/config` file.

`.ssh/config` example:

```
# global settings
ForwardX11        no               # disable X11 forwarding
ForwardX11Trusted no               # do not trust remote X11 clients

# per-host based settings, example for Fram
Host fram                            # alias, you may run "ssh fram" only
	HostName fram.sigma2.no          # actual hostname for Fram
	User my_username                 # replace with your username on Fram
	IdentityFile ~/.ssh/id_rsa_fram  # pointer to your private SSH key
	ForwardX11          yes          # enable X11 forwarding
	ForwardX11Trusted	no           # do not trust remote X11 clients
```


## SHA256 fingerprint

No matter how you login, you will need to confirm that the connection
shall be trusted.  The first time you log in to a machine fia `ssh`, you will
get a message like

    The authenticity of host '<hostname>' can't be established.
    ECDSA key fingerprint is <fingerprint>.
    Are you sure you want to continue connecting (yes/no)?

If the `<fingerprint>` matches the fingerprint of the login machine
you are logging in to (see below), you can confirm by typing `yes` and
press `Enter`.  (Note that the trailing "." is not part of the
fingerprint.)  If the fingerprint does _not_ match, please contact
`support@metacenter.no` immediately.

For the **Fram** login nodes, the ECDSA key fingerprint is

	SHA256:4z8Jipr50TpYTXH/hpAGZVgMAt0zwT9+hz8L3LLrHF8

Some ssh clients will show it as

	MD5:5b:af:a6:1d:94:1c:64:e1:11:54:0e:1f:7d:d2:cd:80

(possibly without the `MD5:` prefix).

For the **Saga** login nodes, it is

	SHA256:qirKlTjO9QSXuCAiuDQeDPqq+jorMFarCW+0qhpaAEA

sometimes shown as

	MD5:13:4e:ae:66:89:0d:24:27:b8:15:87:24:31:ed:32:af

For the **NIRD** login nodes, it is

	SHA256:ZkBvlcu4b5QMf1o9nKzoPHTmSTAzVhogZxKYvNw9N9I

sometimes shown as

	MD5:02:02:cc:9d:c5:b7:43:42:5b:cd:d2:82:09:48:31:e9


## SSHFS 

`fram.sigma2.no` and `login.fram.sigma2.no` are round-robin DNS
entries, every time you use this name the round-robin configuration
will send you to one of the following two login nodes:
`login1.fram.sigma2.no` and `login2.fram.sigma2.no`

When you use `sshfs`, to make sure your authentication is valid, you
should always specify one of the real login nodes above.  You should
not use `login.fram.sigma2.no` or `fram.sigma2.no` in your `sshfs`
command, otherwise you will risk to get your IP address blacklisted,
since your session is authenticad against only one login node not
both.

Similarly, `saga.sigma2.no` and `login.saga.sigma2.no` are round-robin
DNS entries for `login-1.saga.sigma2.no`, `login-2.saga.sigma2.no`.
   

## Poor connection

In case of poor connection to the server (likely from a very remote area),
usually noticeable with X11 forwarding enabled, you may request data 
compression by using the `-C` option.

Please note that the compression uses the CPU to compress-decompress all data
sent over ssh and will actually have negative impact, slow down things on a 
fast network.



[^1]: By default your X authority database is stored in the `~/.Xauthority` file. This file contains records with authorization information used in connecting to the X server.

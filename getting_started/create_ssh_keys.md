# SSH

Some SSH related frequently asked questions are documented down below.
For more in-depth details, other options, please consult the man pages:
`man ssh` and `man ssh_config`.

## Login via ssh keys

To login to a server without typing in your password every time, you can
configure ssh to use public key cryptography. In case you use a linux system
start by generating a pair of keys and saving them in the folder `.ssh`:

```
$ ssh-keygen -t ed25519 -a 100 -f .ssh/id_sigma2
```

Make sure to enter a passphrase to encrypt the key.

To copy and install the public key to the server, for example saga,
we use:
```
$ ssh-copy-id -i ~/.ssh/id_sigma2 username@saga.sigma2.no
```

Using ssh keys has the added benefit that you can avoid having to type your
password every time. `ssh-agent` is program installed on virtually all linux
versions to manage the keys so that you only have to unlock the key once. We
can add the new key with:
```
$ ssh-add ~/.ssh/id_sigma2
```

We recommend you configure your ssh client by adding a section for each Sigma2
system you have access to by editing `.ssh/config`:

```
Host saga
	Hostname saga.sigma2.no
	User myusername
	IdentityFile .ssh/id_sigma2
```

This will let you simply type `ssh saga`, rather than e.g. `ssh
myusername@saga.sigma2.no -i .ssh/id_sigma2`

For more information see [`keygen`](https://www.ssh.com/ssh/keygen/).

## Windows ssh client

In Windows 10 and newer you can now get a fully functional Linux terminal by
[installing WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

This will not only give you a shell with the ssh client, but also let you
install and use all of your favourite Linux software such as `Vim`, `Emacs`,
`nano`, `perl`, `python` and so on.

## X11 forwarding

X11 forwarding should be used with caution due to security implications. Please
note that if someone can read your X authorization database [^1], that person
would be able to access the local X11 display through the forwarded connection.

We suggest switching it on *only* when needed, with the use of options (`-X` or
`-Y`) passed to the `ssh` command. Whenever possible, use `-X` option to mark
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


(ssh_fingerprint)=

## SHA256 fingerprint

No matter how you login, you will need to confirm that the connection shall be
trusted. The first time you log in to a machine via `ssh`, you will get a
message like

```
The authenticity of host '<hostname>' can't be established.
ECDSA key fingerprint is <fingerprint>.
Are you sure you want to continue connecting (yes/no)?
```

If the `<fingerprint>` matches the fingerprint of the login machine you are
logging in to (see below), you can confirm by typing `yes` and press `Enter`.
(Note that the trailing "." is not part of the fingerprint.) If the fingerprint
does _not_ match, please contact `support@metacenter.no` immediately.

For the **Fram** login nodes, the ECDSA key fingerprint is

	SHA256:4z8Jipr50TpYTXH/hpAGZVgMAt0zwT9+hz8L3LLrHF8

Some ssh clients will show it as

	MD5:5b:af:a6:1d:94:1c:64:e1:11:54:0e:1f:7d:d2:cd:80

(possibly without the `MD5:` prefix).

Putty seems to at least sometimes display the ED25519 key instead

	SHA256:PNlam6vviiHF0V7o4cGMmBfXAo5UEgfOtIkFQgXt9bM
	MD5:55:ae:4a:0d:a6:05:76:bb:e7:59:1a:4a:b2:a4:87:2b

For the **Saga** login nodes, it is

	SHA256:qirKlTjO9QSXuCAiuDQeDPqq+jorMFarCW+0qhpaAEA

sometimes shown as

	MD5:13:4e:ae:66:89:0d:24:27:b8:15:87:24:31:ed:32:af

or for ED25519

	SHA256:ryqAxpKDjNLLa5VeUPclQRaZBOIjd2HFgufUEnn4Jrw
	MD5:55:52:eb:a5:c3:a9:18:be:02:15:ea:60:19:d7:5e:06


For the **NIRD** login nodes, it is

	SHA256:ZkBvlcu4b5QMf1o9nKzoPHTmSTAzVhogZxKYvNw9N9I

sometimes shown as

	MD5:02:02:cc:9d:c5:b7:43:42:5b:cd:d2:82:09:48:31:e9

or for ED25519

	SHA256:sI/YOUiasD/yA/g8UMc2Isg4imXs7l8x/QQK01XfaOQ
	MD5:bc:c9:a9:44:ca:b5:cb:53:56:68:02:d1:f1:6a:1a:78


To display all fingerprints for a certain server, you can use the following
command on your local machine (Linux or Mac):

```bash
$ ssh-keygen -lf <(ssh-keyscan **login.nird.sigma2.no** 2>/dev/null)
```


(mosh)=

## Mosh

A description of [Mosh](https://mosh.org):

> Remote terminal application that allows roaming, supports intermittent
> connectivity, and provides intelligent local echo and line editing of user
> keystrokes.

> Mosh is a replacement for interactive SSH terminals. It's more robust and
> responsive, especially over Wi-Fi, cellular, and long-distance links.

Mosh is in many instances a drop-in replacement for `ssh` (and actually utilizes
`ssh` under the hood for establishing a connection). It is recommended to use
Mosh if you connect to Sigma 2 resources from a laptop and want to keep the
connection when roaming on Wi-Fi or putting the laptop to sleep.

Since Mosh uses `ssh` to establish a connection it can use the same `ssh` keys
and configuration, as described above. Which mean that if you created the keys
above and added a section to your `.ssh/config` you can connect by doing the
following

```bash
$ mosh saga # Equivalent to 'mosh <username>@saga.sigma2.no'
```

For more detailed usage information [see the Mosh
homepage](https://mosh.org/#usage).

Unfortunately to support the features of Mosh not everything `ssh` can do is
supported, if you require any of the following you will have to use `ssh`

- `X11` forwarding
- Port forwarding

## SSHFS

`fram.sigma2.no` and `login.fram.sigma2.no` are round-robin DNS
entries, every time you use this name the round-robin configuration
will send you to one of the following two login nodes:
`login1.fram.sigma2.no` and `login2.fram.sigma2.no`

When you use `sshfs`, to make sure your authentication is valid, you should
always specify one of the real login nodes above. You should not use
`login.fram.sigma2.no` or `fram.sigma2.no` in your `sshfs` command, otherwise
you will risk to get your IP address blacklisted, since your session is
authenticated against only one login node not both.

Similarly, `saga.sigma2.no` and `login.saga.sigma2.no` are round-robin DNS
entries for `login-1.saga.sigma2.no`, `login-2.saga.sigma2.no`.


## Poor connection

In case of poor connection to the server (likely from a very remote area),
usually noticeable with X11 forwarding enabled, you may request data compression
by using the `-C` option.

Please note that the compression uses the CPU to compress-decompress all data
sent over ssh and will actually have negative impact, slow down things on a fast
network.



[^1]: By default your X authority database is stored in the `~/.Xauthority`
  file. This file contains records with authorization information used in
  connecting to the X server.

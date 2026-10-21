(ssh)=

# SSH

This page assumes that the reader:
- Has a Sigma2 account, and access to at least one cluster (e.g. `olivia`)
- is working on a Linux machine, a macOS, or a Windows machine with OpenSSH
  installed (default on recent Windows 10+ versions)
  - check in the (linux) terminal with `ssh -V` that you indeed have OpenSSH available

This page is adapted from the very nice documentation written by our
colleagues at Aalto University (Finland):
<https://scicomp.aalto.fi/scicomp/ssh/>.


## What is SSH

SSH is an abbreviation for *secure shell protocol*. It is a protocol used to communicate between two computers over an encrypted connection. When you
log into one of the clusters with `ssh` and read and edit files, type
commands or copy files using `rsync`, then data is transmitted via this
encrypted connection (see also our guide about {ref}`data transfer <data-transfer>`).


## Connecting to a server

```{note}
Using two-factor authentication (2FA) is mandatory when using SSH to connect to NRIS systems. For more details, visit the {ref}`OTP/2FA page <two-factor-authentication>`. Connections between NRIS systems (Saga, Betzy, Olivia, and Nird) are facilitated using SSH key pairs (see page about {ref}`SSH-keypairs<ssh-keypairs>` for more information).
```

When you type `ssh myusername@saga.sigma2.no`, `myusername` is your
username on the remote server, and `saga.sigma2.no` is the server/cluster you
are connecting to:
`````{tabs}
````{group-tab} Saga

```console
$ ssh myusername@saga.sigma2.no
```
````
````{group-tab} Betzy

```console
$ ssh myusername@betzy.sigma2.no
```
````
````{group-tab} Olivia

```console
$ ssh myusername@olivia.sigma2.no
```
````
````{group-tab} NIRD

```console
$ ssh myusername@nird.sigma2.no
```
````
`````

If `myusername` is the same on your computer and the remote server, you can
leave it out:
```console
myusername@localhost:$ ssh saga.sigma2.no
```

## First-time login

When you use SSH to connect to a remote server for the very first time, 
you will be prompted to confirm that the remote server is indeed the one 
you expected to connect to:
```
The authenticity of host 'saga.sigma2.no (2001:700:4a01:10::37)' can't be established.
ED25519 key fingerprint is SHA256:ryqAxpKDjNLLa5VeUPclQRaZBOIjd2HFgufUEnn4Jrw.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

This question helps prevent another server from impersonating the remote
resource and subsequently impersonating you to the real resource. This is not very likely but it is possible, so it's a good idea to double-check the fingerprint and compare it with published {ref}`ssh fingerprints <ssh-fingerprints>`. Note that the trailing period (".") is not part of the fingerprint.

If the fingerprint matches, you can confirm by typing `yes` and press `Enter`. 

```{warning}
If the fingerprints do not match, please {ref}`contact us <support-line>`
immediately.
```

## Jumping through login nodes (*of the same cluster*)

Once logged in, you can easily jump from one login node to another by typing `ssh login-X` (for Saga and Betzy) or `ssh loginX` (for NIRD). Please, replace "X" with the number of the login node you want to access.

The same is valid for when you want to access a specific compute node. However, it is only possible to access compute nodes on which you currently have jobs running.

Since 2FA is now manditory, you will not be able to ssh *to* the login-nodes with keypairs (applies only from host/outside. SSH between login-nodes/clusters uses keypairs). See {ref}`OTP/2FA page <two-factor-authentication>` for more information (and tips).


(ssh-config)=

## Configuring SSH for less typing

To avoid typing in the whole address (e.g. `ssh user@saga.sigma2.no`) everytime you need to access a remote server, you can use a configuration file to create shorter "*aliases* (e.g. `ssh saga`). The file is located at `~/.ssh/config` (create the file if it doesn't exist). Open the file (e.g. `vim ~/.ssh/config`), and add (minimum) the following lines (one file pr local host):

```
Host saga
    User myusername
    Hostname saga.sigma2.no
```

- ***Host** == the alias/shortened name, **User** == your username (or username to use), **Hostname** == full address to remote server*.

After saving the file, instead of typing:
```console
$ ssh myusername@saga.sigma2.no
```

next time you can use the alias to login:
```console
$ ssh saga
```
If you get an error `Bad owner or permissions on /cluster/home/myusername/.ssh/config`, run `chmod 600 ~/.ssh/config`, and try again. You should be able to connect now.

If you have multiple keys for multiple systems, you can *bind* a specific key to a specific host, by adding the line `IdentityFile ~/.ssh/sagaKey` in the config file (replace `sagaKey` with your own key-name). On the example above, this would force every connection to host `saga` to use the private key of the pair `sagaKey`. Also, `rsync`, `scp`, and any other tool that uses `ssh` under the hood will understand these shortcuts. Search the web for more useful configuration options for automation.

## SSH client on Windows

In Windows 10 and newer you can now get a fully functional Linux terminal by [installing WSL](https://learn.microsoft.com/en-us/windows/wsl/install). Another alternative is to use the [Windows SSH Client](https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh) directly.

(x11-forwarding)=

## X11 forwarding

X11 forwarding is a method to send the graphical screen output from the remote server to your local computer.

X11 forwarding should be used with caution due to security implications.
Please note that if someone can read your X authorization database, that
person would be able to access the local X11 display through the forwarded
connection. By default, your X authority database is stored in the
`~/.Xauthority` file. This file contains records with authorization
information used in connecting to the X server.

We suggest switching it on *only* when needed, using the options (`-X` or `-Y`) passed to the `ssh` command. Whenever possible, use the `-X` option to mark remote X11 clients untrusted.

In some cases, `-X` may fail to work. When this happens, you can either use the `-Y` option or set `ForwardX11Trusted` to `yes` in your SSH configuration  file. Be aware that this will grant remote X11 clients full access to the original X11 display.

Alternatively, if X11 forwarding is always needed, you can configure it on a per-host basis in your `.ssh/config` file:
```
# global settings
ForwardX11        no               # disable X11 forwarding
ForwardX11Trusted no               # do not trust remote X11 clients

# per-host settings, example for Saga
Host saga                            # alias, you may run "ssh saga" only
    HostName saga.sigma2.no          # actual hostname for Saga
    User my_username                 # replace with your username on Saga
    IdentityFile ~/.ssh/id_rsa_saga  # pointer to your private SSH key
    ForwardX11          yes          # enable X11 forwarding
    ForwardX11Trusted   no           # do not trust remote X11 clients
```


## SSHFS

[SSHFS](https://github.com/libfuse/sshfs) allows you to mount a remote
file system using SFTP.

If you wish to use SSHFS, please note that `saga.sigma2.no`,
`login.saga.sigma2.no`, and addresses for other clusters are round-robin
entries. This means that every time you log in, you might end up on a
different actual login node (e.g., `login-1.saga.sigma2.no` or
`login-3.saga.sigma2.no`). This is done to balance the load between the login-nodes.

When using `sshfs`, always specify one of the actual login nodes 
(e.g., `login-1.saga.sigma2.no`), not the "front-ends". Otherwise your session will authenticate against only one login node, which can result in your IP address being blacklisted.

**NB**: After enabling two-factor authentication, this is only available on port 12, see {ref}`OTP/2FA page<two-factor-authentication>`.

## Compressing data for poor connections

In case of poor connection to the server, likely from a very remote area, and usually noticeable with X11 forwarding enabled, you may request data
compression by using the `-C` option. Please note that compression relies on the CPU to compress and decompress all data. If you are on a fast network, this will have a negative impact on your bandwidth.


## SSH over breaking connections

If you experience intermittent connectivity (e.g., on Wi-Fi, cellular,
long-distance links) and SSH disconnects frequently, you can configure SSH  to attempt to keep the connection alive by adding this into your
`~/.ssh/config` file:
```
Host saga
    Hostname login.saga.sigma2.no
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

For even more unstable connections, try [Mosh (mobileshell)](https://mosh.org/). Mosh is in many instances a drop-in replacement for `ssh` (and actually utilizes `ssh` under the hood for establishing a connection). It is recommended to use Mosh if you connect from a laptop and want to keep the connection when roaming on Wi-Fi or putting the laptop to sleep.

(ssh_errors)=

# Common SSH errors

Below is a description and solution for commonly receieved errors when attempting ssh connections.

## WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!

The SSH connection was working fine until one day the following message
appeared:

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:XXX.
Please contact your system administrator.
Add correct host key in /home/username/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /home/username/.ssh/known_hosts:13
  remove with:
  ssh-keygen -f "/home/username/.ssh/known_hosts" -R saga.sigma2.no
ED25519 host key for saga.sigma2.no has changed and you have requested strict checking.
Host key verification failed.
```

It may be frightening at first, but it usually means that the SSH Keys from the server have changed, which is common after a system upgrade (so, take a look at our [OpsLog](https://opslog.sigma2.no/) for more information).

The fix is already in the message itself and, in this example, you just have to locate the file `known_hosts` inside `/home/username/.ssh/` and delete the specified line (here line `13`).

**NOTE:**
- The number at the end indicates where the problem lies.
- The path will be different according to the operating system you are running.
- Also, on Linux, having a folder starting with `.` means it is a hidden folder.

Also, if you are familiar with Linux terminal, running the suggested command also has the same effect:
`ssh-keygen -f "/home/username/.ssh/known_hosts" -R saga.sigma2.no`

After completing the steps above, try logging in again and accept the new
fingerprint (to verify it is the correct one, see {ref}`ssh fingerprints <ssh-fingerprints>`.

## References

- <https://scicomp.aalto.fi/scicomp/ssh/> - inspiration for this page
- <https://www.mn.uio.no/geo/english/services/it/help/using-linux/ssh-tips-and-tricks.html> - long-form guide
- <https://blog.0xbadc0de.be/archives/300> - long-form guide
- <https://www.phcomp.co.uk/Tutorials/Unix-And-Linux/ssh-passwordless-login.html>
- <https://en.wikipedia.org/wiki/OpenSSH>
- <https://linuxize.com/post/ssh-command-in-linux/#how-to-use-the-ssh-command>
- <https://linuxize.com/post/how-to-setup-passwordless-ssh-login/>
- <https://infosec.mozilla.org/guidelines/openssh>
- <https://www.ssh.com/ssh/> - commercial site

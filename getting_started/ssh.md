(ssh)=

# SSH

This page assumes that the reader:
- is working on a Linux machine, a macOS or a Windows machine with OpenSSH
  installed (default on recent Windows 10+ versions)
- check in the terminal with `ssh -V` that you indeed have OpenSSH available
- has an account on the server of interest

This page is adapted from the very nice documentation written by our
colleagues at Aalto University (Finland):
<https://scicomp.aalto.fi/scicomp/ssh/>.


## What is SSH

SSH is an abbreviation for *secure shell protocol*. It is a protocol to
communicate data between two computers over an encrypted connection. When you
log into one of the clusters with `ssh` and read and edit files and type
commands or copy files using `scp`, then data is transmitted via this
encrypted connection.


## Connecting to a server

When you type `ssh myusername@saga.sigma2.no`, then `myusername` is your
username on the remote server and `saga.sigma2.no` is the server/cluster you
are connecting to.

If `myusername` is the same on your computer and the remote server, you can
leave it out:
```console
$ ssh saga.sigma2.no
```

{ref}`Further below <ssh-config>` we will show how we can configure SSH so that we
don't have to type the same lengthy command every time


## First-time login

When you ssh to a remote server for the very first time, you will be prompted
to affirm that the remote server is indeed the one you expected to connect to:
```
The authenticity of host 'saga.sigma2.no (2001:700:4a01:10::37)' can't be established.
ED25519 key fingerprint is SHA256:ryqAxpKDjNLLa5VeUPclQRaZBOIjd2HFgufUEnn4Jrw.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

This question is to prevent some other server impersonating the remote
resource and subsequently impersonating you to the real resource. This is not
very likely but it is possible, therefore it's a good idea to double check the
fingerprint and compare it with published {ref}`ssh-fingerprints`.

If the `<fingerprint>` matches the fingerprint of the login machine you are
logging in to (see below), you can confirm by typing `yes` and press `Enter`.
Note that the trailing "." is not part of the fingerprint.

```{warning}
If the fingerprints do not match, please {ref}`contact us <support-line>`
immediately.
```

(ssh-config)=

## Configuring SSH for less typing

Remembering the full settings list for the server you are working on each time
you log in can be tedious: the username is the same every time, the server is
the same every time, ... **There is a better way!**

A configuration file allows you to store your preferred settings and map them
to much simpler login commands.

Create or edit (if it already exists) the file `~/.ssh/config`.
Here is an example entry for one of our clusters:
```
Host saga
    User myusername
    Hostname saga.sigma2.no
```

Now instead of:
```console
$ ssh myusername@saga.sigma2.no
```

I can type:
```console
$ ssh saga
```

Also `scp` and `rsync` and any other tool that uses `ssh` under the hood will
understand these shortcuts. There is a lot more that can be configured. Search
the web for more examples if you are interested.


## Using SSH keys instead of passwords

It's boring to type the password every time, especially if you regularly have
multiple sessions open simultaneously (there exist also other tools to help
with that). The tedium of typing it 20-50 times each day could motivate some
to make the password very short or very memorable, thus reducing security.
See also [the relevant XKCD comic](https://xkcd.com/936/).

**There is a better way**: using SSH key pairs. This is not only less tedious
(you will only have to type a passphrase typically once per day), but also
more secure (we will explain why).

An SSH key pair consists of a private key (which you never share with anybody)
and a public key (which you can share with others without problems). Others
can then encrypt messages to you using the public key and you can decrypt them
using your private key. Others can only encrypt. Only you can decrypt.

The private key is a file on your computer. Also the public key is a different
file on your computer. Anybody who has access to your private key can read
data between you and remote servers and impersonate you to the remote servers.

One way to visualize this is to image the public key to be a box where
somebody can put a secret message. Anybody can put something into a box and
close the box and send the box to you, but only you have the key to open it
(private key).

To make sure that your private key (file) does not fall into the wrong hands,
it is custom and **recommended to encrypt it with a passphrase**. Having the
private key "encrypted" with an empty passphrase is possible but it is the
equivalent of leaving your house key under the door mat or the equivalent of
having a bank card without any pin.

**Why are SSH key pairs more secure than using a password?** There is still the
passphrase to unlock the private key so why is this easier and better?  We
will show later how it is easier but it is more secure since the passphrase is
never communicated to the remote server: it stays on your computer.  When the
remote server is authenticating you, it encrypts a large number and sends it
encrypted to you and asks you to decrypt it and send the decrypted number back
and then compares the two. If they match, the remote server knows that you are
you and from there on can trust you for the duration of the session. No
password or passphrase needs to leave your computer over the network.


### Generating a new SSH key pair

While there are many options for the key generation program ``ssh-keygen``, here are the four main ones:
- `-t`: The encryption type used to make the unique key pair.
- `-b`: The number of key bits.
- `-f`: Filename of key.
- `-C`: Comment on what the key is for.
- `-a`: Number of key derivation function rounds. Default is 16. The higher,
    the longer it takes to verify the passphrase but also the better
    protection against brute-force password cracking.

We recommend the following command to create a key pair:
```console
$ ssh-keygen -t ed25519 -a 100
```

After running this command in the terminal, you will be prompted to enter a
passphrase.  **Make sure to enter a passphrase to encrypt the key!** A private
key with an empty passphrase can be used by anybody who gets access to your
private key file. Never share it with anybody!

Upon confirming the password, you will be presented with the key fingerprint
as both a SHA256 hex string as well as randomart image. Your new key-pair
should be found in the hidden `~/.ssh` directory.  If you ran the command
above, you will find there `id_ed25519` (private key, never share it) and
`id_ed25519.pub` (public key, no problem to share).


### Copy public key to server

In order to use your key pair to login to the remote server, you first need to
securely copy the desired *public key* to the machine with ``ssh-copy-id``.
The script will also add the key to the ``~/.ssh/authorized_keys`` file on the
server. You will be prompted to enter your *password* (not the *passphrase*
associated with the private key) to initiate the secure copy of the file.

To copy and install the public key to the server, for example Saga, we use:
```console
$ ssh-copy-id -i ~/.ssh/id_sigma2 myusername@saga.sigma2.no
```

This command creates the directory `~/.ssh` on the target machine
(`saga.sigma2.no` in the example above) if it did not exist yet.  When created
by OpenSSH (e.g. through `ssh-copy-id`), the directory gets the required
strict permissions `0700`, which may be different from the shell's
file-creation mask returned by `umask -S`.  You can check the permissions by
running `ls -ld ~/.ssh` on Saga, and change the permissions to `0700` with the
command `chmod 0700 ~/.ssh`.

Once the public key has been copied to the remote server, you can login using
the SSH key pair. Try it. **It should now ask you for your passphrase and not
for the password.**

This approach works not only for our clusters but also for services like
GitHub or GitLab. But let's focus here on clusters.

````{admonition} Help! It still asks for a password!

In this case, debug with:
```console
$ ssh -v myusername@saga.sigma2.no
```

Instead of `-v` you can also try `-vv` or `-vvv` for more verbose output.
Study the output and try to figure out what goes wrong. Does it try the key
pair you created?
````


### How many key pairs should I create?

We recommend to create a key pair per hardware device. Not a key pair per
remote server.

In other words, if you have a laptop and a desktop and want to authenticate to
4 different servers, create a key pair on the laptop and another one on the
desktop, and upload both public keys to all 4 remote servers.

The motivation to have one key pair per hardware device is that if you lose
your hardware device (e.g. laptop) or it gets stolen, you know which key to
revoke access from.


### Using the OpenSSH authentication agent

Further up we motivated that we don't want to type the password every time
many times a day. Now we instead need to type the private key passphrase every
time so it feels like this was not a win. But again there is a better way: To
avoid having to type the decryption passphrase, the *private key* needs to be
added to the ``ssh-agent`` with the command:
```console
$ ssh-add
```

On macOS, use this instead:
```console
$ ssh-add --apple-use-keychain
```

If you are unsure whether an `ssh-agent` process is running on your machine,
`ps -C ssh-agent` will tell you if there is. To start a new agent, use:
```console
$ eval $(ssh-agent)
```

Once the password is added, you can ssh into the remote server as normal but
will immediately be connected without any further prompts.


## SSH client on Windows

In Windows 10 and newer you can now get a fully functional Linux terminal by
[installing WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

Yet another alternative is to use the [Windows SSH Client](https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh) directly.


(x11-forwarding)=

## X11 forwarding

X11 forwarding is a method to send the graphical screen output from the remote
server to your local computer.

X11 forwarding should be used with caution due to security implications.
Please note that if someone can read your X authorization database, that
person would be able to access the local X11 display through the forwarded
connection.  By default your X authority database is stored in the
`~/.Xauthority` file. This file contains records with authorization
information used in connecting to the X server.

We suggest switching it on *only* when needed, with the use of options (`-X`
or `-Y`) passed to the `ssh` command. Whenever possible, use `-X` option to
mark remote X11 clients untrusted.

In some cases `-X` will fail to work and either the use of `-Y` option or
setting `ForwardX11Trusted` in your SSH configuration file to "yes" is required. In
this case remote X11 clients will have full access to the original X11 display.

Alternatively, if X11 forwarding is always needed, you can configure it on a
per-host basis in your `.ssh/config` file:
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
    ForwardX11Trusted   no           # do not trust remote X11 clients
```


## SSHFS

[SSHFS](https://github.com/libfuse/sshfs) allows you to mount a remote
file system using SFTP.

If you wish to use SSHFS, please note that `fram.sigma2.no`,
`login.fram.sigma2.no`, and addresses for other clusters are round-robin
entries. This means that every time you log in, you might end up on a
different actual login node (e.g. `login-1.fram.sigma2.no` or
`login-2.fram.sigma2.no`). This is done to balance load between login nodes.

When you use `sshfs`, you should always specify one of the actual login nodes,
not the "front-ends", otherwise you risk getting your IP address blacklisted,
since your session is authenticated against only one actual login node and not
the other login nodes.


## Compressing data for poor connections

In case of poor connection to the server, likely from a very remote area and
usually noticeable with X11 forwarding enabled, you may request data
compression by using the `-C` option.

Please note that the compression uses the CPU to compress and decompress all data. If you are on a fast network, then this option will have a negative impact on your bandwidth.


## SSH over breaking connections

If you experience intermittent connectivity when on Wi-Fi, cellular, and
long-distance links and get frustrated with SSH losing connection and you
having to open a new terminal every time, have a look at [Mosh (mobile
shell)](https://mosh.org/).

Mosh is in many instances a drop-in replacement for `ssh` (and actually
utilizes `ssh` under the hood for establishing a connection). It is
recommended to use Mosh if you connect from a laptop and want to keep the
connection when roaming on Wi-Fi or putting the laptop to sleep.


## References

- <https://scicomp.aalto.fi/scicomp/ssh/> - inspiration for this page
- <https://www.mn.uio.no/geo/english/services/it/help/using-linux/ssh-tips-and-tricks.html> - long-form guide
- <https://blog.0xbadc0de.be/archives/300> - long-form guide
- <https://www.phcomp.co.uk/Tutorials/Unix-And-Linux/ssh-passwordless-login.html>
- <https://en.wikibooks.org/wiki/OpenSSH/>
- <https://linuxize.com/post/ssh-command-in-linux/#how-to-use-the-ssh-command>
- <https://linuxize.com/post/how-to-setup-passwordless-ssh-login/>
- <https://hpc-uit.readthedocs.io/en/latest/account/login.html>
- <https://infosec.mozilla.org/guidelines/openssh>
- <https://www.ssh.com/ssh/> - commercial site

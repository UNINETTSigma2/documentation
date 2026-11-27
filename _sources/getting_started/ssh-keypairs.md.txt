(ssh-keypairs)=

# SSH-keypairs

**NB**: *After the mandatory implementation of two-factor authentication on Sigma2 user accounts, SSH keys alone only work for transferring files via port 12, and for interconnection **between** our clusters. Password must be used with OTP for regular access **to** the clusters*. Check the {ref}`OTP/2FA page<two-factor-authentication>`. This section is therefore valid for connecting between hosts that can accept keypair (only) options (on our clusters, only *interconnection* between: `betzy`, `nird`, `olivia`, and `saga`), and for the filetransfer process mentioned above.

*For the examples in this section, we will use `saga` as host, `olivia` as destination, the keypair `id_ed25519` and `id_ed25519.pub`, and `myuser` as our Sigma2 username*. 

## Generating and using a new SSH key pair

First and foremost, you will need to create an ssh key-pair on the local host. While there are many options for the key generation program ``ssh-keygen``, here are the main ones:
- `-t`: The encryption type used to make the unique key pair.
- `-b`: The number of key bits. (Ed25519 is fixed length and ignores this)
- `-f`: Filename of key.
- `-C`: Comment on what the key is for.
- `-a`: Number of key derivation function rounds. Default is 16. The higher, the longer it takes to verify the passphrase, but also the better protection against brute-force password cracking.

The Ed25519 key type is preferable to RSA keys. They are shorter and easier to identify visually. Always use comments to distinguish between keys.

**We recommend the following command to create a key pair**:
```console
ssh-keygen -t ed25519 -a 100 -f "$HOME/.ssh/id_ed25519" -C "$(whoami)@$(hostname)-$(date -I)"
```

After running this command in the terminal, you will be prompted to enter apassphrase.  **Make sure to enter a passphrase to encrypt the key!** A private key with an empty passphrase can be used by anybody who gets access to your private key file. Never share it with anybody!

Upon confirming the passphrase, you will be presented with the key fingerprint as both a SHA256 hex string and as a randomart image. Your new key pair will be saved in the hidden `~/.ssh` directory.  If you ran the command above, you will find the following files there: `id_ed25519` (private key, never share it) and `id_ed25519.pub` (public key, no problem to share).

### Binding keys to hosts

This section is for the ssh configuration file `~/.ssh/config`. See section on {ref}`config files<ssh-config>` for more information. Once you have created a keypair, add a line to your `config` file specifying that "*all connection attempts to HOST should try KEY*": `IdentityFile ~/.ssh/id_ed25519`. This will ensure that SSH tries `id_ed25519` pair to authenticate to the given host (first, if this fails it will go to the alternative options available on the remote host).

### Using keypairs

In order to use your key pair to log in to the remote server, you first need to securely copy the *public key* to the remote host with `ssh-copy-id`. The script will also add your public key to the `~/.ssh/authorized_keys` file on the remote host (next time you login to the remote host your key will be recognized). You will be prompted to authenticate with your  password + OTP (the username connecting to the remote host) to initiate the secure copy of the file.

To copy and install the public key to the server (here from `saga` to `olivia`), use:
```console
myuser@login-1.saga..:$ ssh-copy-id -i ~/.ssh/id_ed25519 myuser@olivia.sigma2.no
```
You will now be asked to authenticate using your OTP + password. Once you've done this, you should see the following message:
````
Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'myuser@olivia.sigma2.no'"
and check to make sure that only the key(s) you wanted were added.
```` 

**NB**: To spread load over all the login nodes of a given cluster (Saga, Betzy, NIRD, Olivia), e.g. `ssh saga.sigma2.no` will lead the user to 1 of the 5
available login nodes of saga (this is seen by the `user@login-X.saga...` on the shell). In the example above, your key would be copied to all login nodes on olivia (`uanX`).

This command creates the file `authorized_keys` in the directory `~/.ssh` on the target machine if it did not exist yet, or adds the new key to the existing file (`olivia.sigma2.no:/cluster/home/myuser/.ssh/authorized_keys` in the example above). When created by OpenSSH (e.g. through `ssh-copy-id`), the directory gets the required strict permissions `0700`, which may be different from the shell's file-creation mask returned by `umask -S`.  You can check the permissions by running `ls -ld ~/.ssh` on Saga, and change the permissions to `0700` with the command `chmod 0700 ~/.ssh`.

Once the public key has been copied to the remote server, you can log in using the SSH key pair (if it isn't defined in the `config` file, you can specify with `-i` parameter): `ssh -i ~/.ssh/sagaKey1 myuser@olivia.sigma2.no`. This will now ask for your **pasphrase** which you set when creating the keypair.

**This is a standard approach that works not only for our clusters but also for other servers/clients, as well as services like GitHub or GitLab.**

- **Why are SSH key pairs more secure than using a password?** There is still the passphrase to unlock the private key, so why is this easier and better? We will show later how it is easier, but it is more secure because **the passphrase is never communicated to the remote server**: it stays on your computer. When the remote server is authenticating you, it encrypts a large number and sends it encrypted to you, asks you to decrypt it and send the decrypted number back, and then compares the two. If they match, the remote server knows that you are you and from there on can trust you for the duration of the session. No password or passphrase needs to leave your computer over the network.


````{admonition} Help! It still asks for a password!

In this case, debug with:
```console
$ ssh -v myusername@saga.sigma2.no
```

Instead of `-v` you can also try `-vv` or `-vvv` for more verbose output. When {ref}`contacting us <support-line>`, remember to include this output in the request. *You should also check the output and try to figure out what goes wrong. Does it try the key pair you created?* 
````


## How many key pairs should I create?

We recommend creating a key pair for each hardware device, *not* a key pair for each remote server. In other words, if you have a laptop and a desktop and want to authenticate to 4 different servers, create a key pair on the laptop and another one on the desktop, and upload both public keys to all 4 remote servers. The motivation is that if you lose your hardware device (e.g., laptop) or it gets stolen, you know exactly which key to revoke access from (in the example above, `sagaKey` is for connecting *from* `saga`).


## Using the OpenSSH authentication agent

Earlier we mentioned that *it's easier to use ssh-keys rather than passwords*. This is because we can add the **private key** to the `ssh-agent` on our host machine. This will require *one* authentication, and keep that authentication *alive* as a process until the machine is restarted (or process is killed). After this, whenever you use that ssh keypair you will not need to re-write the passphrase. 
This requires that there is a `ssh-agent` process (check with `ps aux | grep ssh-agent`, shows also the user associated with the given process) on the host. If you don't see one (associated to your user in a multi-user host), you can create this by running `eval $(ssh-agent)`. Confirm that *your* process is now alive, and add your key using the following command (*remember to replace `sagaKey1` with your keyName):

`````{tabs}
````{group-tab} Linux

```console
$ ssh-add ~/.ssh/sagaKey1
```
````
````{group-tab} Windows
Remember to have the service "OpenSSH Authentication Agent" enabled
and starting automatically.

```console
$ ssh-add ~/.ssh/sagaKey1
```
````
````{group-tab} macOS

```console
$ ssh-add --apple-use-keychain ~/.ssh/sagaKey1
```
````
`````

Once the password is added, you can SSH into the remote server as normal, and you will be immediately connected without any further prompts.

Typically, you use `ssh-add` once per day (e.g. until terminal/command prompt window is closed), after which you can use `ssh` and `rsync` as often as you like without re-authenticating. You can check if there are any active processes running with `ps aux | grep ssh` (look for `ssh-agent`).


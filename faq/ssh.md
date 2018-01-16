# SSH

Some SSH related frequently asked questions are documented down below.
For more in-depth details, other options, please consult the man pages: 
`man ssh` and `man ssh_config`.

## X11 forwarding

X11 forwarding should be used with caution due to security implications. Please
note that if someone can read your X authorization database, that person would 
be able to access the local X11 display through the forwarded connection.

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

# per-host based settings
Host fram                          # alias, you may run "ssh fram" only
	HostName fram.sigma2.no          # actual hostname for Fram
	User my_username                 # replace with your username on Fram
	IdentityFile ~/.ssh/id_rsa_fram  # pointer to your private SSH key
	ForwardX11        yes            # enable X11 forwarding
	ForwardX11Trusted	no             # do not trust remote X11 clients
```

## Poor connection

In case of poor connection to the server (likely from a very remote area), 
usually noticeable with X11 forwarding enabled, you may request data 
compression by using the `-C` option.

Please note that the compression uses the CPU to compress-decompress all data
sent over ssh and will actually have negative impact, slow down things on a 
fast network.

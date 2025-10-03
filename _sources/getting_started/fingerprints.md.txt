---
orphan: true
---

(ssh-fingerprints)=

# Key fingerprints of our systems

The following overview displays the different keys (depending on which key type
you use to connect), both as `MD5` and `SHA256`, for all systems.

```{warning}
If the fingerprints do not match what is presented to you upon first-time
login, please {ref}`contact us <support-line>` immediately.
```


## fram.sigma2.no

- ED25519:
  - `MD5:5f:15:92:e1:22:74:69:68:6c:1c:27:f5:a5:b1:76:3f`
  - `SHA256:hLb9eJdGcTT2PHoWamc/+06LlF+vgcnyfvFEqh60cT8`
- ECDSA:
  - `MD5:f7:83:b9:99:b4:6e:aa:af:21:d6:ae:de:b2:14:ab:6a`
  - `SHA256:Nd1Sqijzb1qx26wjCwThChdKz9iPMhJQ9zgVKcD+P5g`
- RSA:
  - `MD5:05:d0:0e:fa:cb:72:c0:03:cb:8f:d0:b4:dc:09:04:4e`
  - `SHA256:Cq5Vt82wQAAhMu4q05L3gmB4QeW1POpNNKgTIP8A2f4`

## saga.sigma2.no

- ED25519:
  - `MD5:2b:c2:ce:c0:f1:b8:0a:95:ec:db:b4:f3:fb:ee:e9:70`
  - `SHA256:YOkZ1uudXrFmaigdnpZ64z497ZccNhdZe/abFkDXOH8`
- ECDSA:
  - `MD5:13:4e:ae:66:89:0d:24:27:b8:15:87:24:31:ed:32:af`
  - `SHA256:qirKlTjO9QSXuCAiuDQeDPqq+jorMFarCW+0qhpaAEA`
- RSA:
  - `MD5:61:e4:49:4b:4e:00:14:2d:9d:b9:ac:99:c2:16:e6:ab`
  - `SHA256:mY+Po9LKAlZGzMRHUmq1abrSOohifdN7+5VUmRTW4tE`


## betzy.sigma2.no

- ED25519:
  - `MD5:de:75:8c:93:40:f6:32:94:b6:bd:47:43:62:a5:1a:58`
  - `SHA256:7M0HDP163k9fOUeZq3KtzLdjISE9Kq/gVygCpyrZPDQ`
- ECDSA:
  - `MD5:37:da:0d:cd:fe:66:47:71:3f:08:59:d7:bb:76:ec:cc`
  - `SHA256:l0adSAGOHM4CNOqxvBNh5Laf+PlDSXQiargVoG/cue4`
- RSA:
  - `MD5:f6:a9:4e:a7:f6:1e:10:5c:01:e7:44:ac:34:4d:4b:b4`
  - `SHA256:wSirru+JTpcAZKQe/u6jLRj3kVCccNNUWU2PxzgbebM`


## login.nird.sigma2.no

- ED25519:
  - `MD5:c4:23:90:52:eb:d9:eb:e5:41:0d:ef:4d:ac:78:2c:db`
  - `SHA256:A8gq7aiQHoK4QzRi1hMpLNbo40ZTZxlGfDCpDWZy/ZQ`
- ECDSA:
  - `MD5:78:ea:cb:f7:f0:6b:02:55:17:0c:b1:5f:de:2a:3e:78`
  - `SHA256:lawnWA5fHTX64XB8OU0WUrQu/dCtFgCfvMC+i/zBCrI`
- RSA:
  - `MD5:54:c9:b7:71:22:9e:bd:e9:ad:5c:18:fe:7b:41:e7:01`
  - `SHA256:xpJZ+XiY4oy3df/R5/LN8i30Z5/EiYo6YSCUQdKkQ/U`


## Display all fingerprints for a certain server

To display all fingerprints for a certain server, you can use the following
command on your local machine (Linux or macOS):

```console
$ ssh-keyscan login.nird.sigma2.no | ssh-keygen -l -f - -E md5
$ ssh-keyscan login.nird.sigma2.no | ssh-keygen -l -f - -E sha256
```

# Common SSH errors when keys have changed

Please, take a look at this {ref}`page <ssh_errors>`

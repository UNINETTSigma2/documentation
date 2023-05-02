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
  - `MD5:5c:49:52:68:a1:aa:d7:dd:3e:71:4a:3b:fd:6f:ef:7b`
  - `SHA256:m3tuW22Y3K+wztKWEOKl8vXxfD2aJxknveyQCsxu+a8`
- ECDSA:
  - `MD5:5b:af:a6:1d:94:1c:64:e1:11:54:0e:1f:7d:d2:cd:80`
  - `SHA256:4z8Jipr50TpYTXH/hpAGZVgMAt0zwT9+hz8L3LLrHF8`
- RSA:
  - `MD5:6d:3b:72:31:c8:38:9c:c5:c9:a6:8a:aa:ee:50:38:da`
  - `SHA256:0FqUvnjU5OAXkmx2j1U2Z8rxmwq/Sm12lN+i+HrqnaQ`


## saga.sigma2.no

- ED25519:
  - `MD5:55:52:eb:a5:c3:a9:18:be:02:15:ea:60:19:d7:5e:06`
  - `SHA256:ryqAxpKDjNLLa5VeUPclQRaZBOIjd2HFgufUEnn4Jrw`
- ECDSA:
  - `MD5:13:4e:ae:66:89:0d:24:27:b8:15:87:24:31:ed:32:af`
  - `SHA256:qirKlTjO9QSXuCAiuDQeDPqq+jorMFarCW+0qhpaAEA`
- RSA:
  - `MD5:83:5e:c5:c3:95:dc:9e:b9:34:87:f6:df:4b:74:04:6b`
  - `SHA256:WgRP8okUDv2j8SwDxj7ZoNRQDOlJwVtvRqVf1SzXgdU`


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

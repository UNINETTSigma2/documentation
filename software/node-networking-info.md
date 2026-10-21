---
orphan: true
---

(nodes-network)=

# Networking on nodes

Below is information regarding network on nodes. This applies both for general access to the Internet, as well as information on license servers.
If you have any questions/corrections/updates on the information below, please {ref}`contact support<support-line>`.

## Login nodes:
       
- All login nodes have direct outbound and opened access to whole internet and all protocols via nodes' public IP and appropriate gateway

- The dual stack is available on these nodes (meaning both IPv4 and IPv6 connectivities are available independently)


## Compute nodes:

- Compute nodes have full opened access to license servers needed by various software

    - if a user can't contact a license server, from compute node they could try to run `nmap -p 1234 mylicenseserver.com` and 
      `nc -vz mylecenseserver.com 1234`, where "1234" is the port mylicenseserver.com should listen on
    
    - if output of the command above yields anything else than open state, there are 2 potential problems - see below.


## 1. The license server is not configured properly:

Please ensure it's listening on the right port and it's opened for incoming connection from the following ip ranges:

| **Cluster Name** | **Public login** | **Public service** |
| -- | -- | -- |
| Betzy  | 158.36.141.144/28<br>2001:700:4a01:25::/64 | 158.36.154.0/28<br>2001:700:4a01:23::/64   |
| Olivia | 158.38.122.192/27<br>2001:700:570f:50::/64 | 158.38.122.224/27<br>2001:700:570f:51::/64 |
| Saga   | 158.36.42.32/28<br>2001:700:4a01:10::/64   | 158.36.42.48/28<br>2001:700:4a01:21::/64   |


If this is in place and user still can't contact the license server from a compute node, please contact {ref}`contact support<support-line>` and we will add the license server on our side. <br>

## 2. You may need to use a proxy

Compute nodes have limited access to the Internet. This means that a user is (only) allowed to reach `http`/`https` services via a proxy server. Users can check the proxy server setup by running `env | grep -i proxy`, which should output the name of proxy server. 
If you do not get any output, you can run the following commands to set the proxy:
- **Betzy**:  `export https_proxy=http://admin.betzy.sigma2.no:3128/` and `export http_proxy=http://admin.betzy.sigma2.no:3128/`.
- **Olivia**: `export https_proxy=http://10.63.2.48:3128/` and `export http_proxy=http://10.63.2.48:3128/`.
- **Saga**:   `export https_proxy=http://proxy.saga:3128/` and `export http_proxy=http://proxy.saga:3128/`

Now run `env | grep -i proxy` again, and you should see both proxies. If the external license server has allowed communication from the addresses above, you should get a "success" message when running `nc -vz mylecenseserver.com 1234`.

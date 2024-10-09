# Login nodes:
       
- All login nodes have direct outbound and opened access to whole internet and all protocols via nodes' public IP and appropriate gateway

- The dual stack is available on these nodes (meaning both IPv4 and IPv6 connectivities are available independently)


# Compute nodes:

- Compute nodes have full opened access to license servers needed by various software

    - if a user can't contact a license server, from compute node they could try to run `nmap -p 1234 mylicenseserver.com` where "1234" is the port mylicenseserver.com should listen on
    
    - if output of the command above yields anything else than open state, there are 2 potential problems:


## The license server is not configured properly:

Please ensure it's listening on the right port and it's opened for incoming connection from the following ip ranges:

```
SAGA    158.36.42.32/28	    2001:700:4a01:10::/64   158.36.42.48/28 	2001:700:4a01:21::/64
FRAM	158.39.114.64/27	2001:700:4a00:10::/64
BETZY	158.36.141.144/28	2001:700:4a01:25::/64   158.36.154.0/28		2001:700:4a01:23::/64
```

If this is in place and user still can't contact the license server from a compute node, please contact `support@nris.no` and we will add the license server on our side.


## You may need to use a proxy

Compute nodes have limited access whole internet, it means a uses is allowed to reach http and https services via a proxy server. Users can check the proxy server setup by running `env | grep -i proxy` which should spit out the name of proxy server

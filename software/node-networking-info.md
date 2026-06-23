:orphan:

(nodes-network)=

# Login nodes:
       
- All login nodes have direct outbound and opened access to whole internet and all protocols via nodes' public IP and appropriate gateway

- The dual stack is available on these nodes (meaning both IPv4 and IPv6 connectivities are available independently)


# Compute nodes:

- Compute nodes have full opened access to license servers needed by various software

    - if a user can't contact a license server, from compute node they could try to run `nmap -p 1234 mylicenseserver.com` where "1234" is the port mylicenseserver.com should listen on
    
    - if output of the command above yields anything else than open state, there are 2 potential problems - see below.


## 1. The license server is not configured properly:

Please ensure it's listening on the right port and it's opened for incoming connection from the following ip ranges:
<table>
<table style="width:100%">
<thead>
<tr>
<th> Cluster Name </th>
<th> Public login </th>
<th> Public service </th>
</tr>
</thead>
<tbody>
<tr>
<td>SAGA</td>
<td> 158.36.42.32/28<br>2001:700:4a01:10::/64</td>
<td>158.36.42.48/28<br>2001:700:4a01:21::/64</td>
</tr>
<tr>
<td>BETZY</td>
<td>158.36.141.144/28<br>2001:700:4a01:25::/64</td>
<td>158.36.154.0/28<br>2001:700:4a01:23::/64</td>
</tr>
<tr>
<td>OLIVIA</td>
<td>158.38.122.192/27<br>2001:700:570f:50::/64</td>
<td>158.38.122.224/27<br>2001:700:570f:51::/64</td>
</tr>
</tbody>
</table>
<br>
If this is in place and user still can't contact the license server from a compute node, please contact `support@nris.no` and we will add the license server on our side. <br>

## 2. You may need to use a proxy

Compute nodes have limited access whole internet, it means a uses is allowed to reach http and https services via a proxy server. Users can check the proxy server setup by running `env | grep -i proxy` which should spit out the name of proxy server

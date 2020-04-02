#
# Create ssh tunnel to fram from the outside world.
# Use localhost:5901 as the server address in the VNC client.
#

# Change USERNAME to the username on fram.
plink.exe -L 5901:localhost:5901 USERNAME@desktop.fram.sigma2.no




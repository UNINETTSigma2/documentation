(apptainer-cache)=
# Apptainer Cache
Apptainer handles images as standard `.sif` files on your disk. To save the network bandwidth and the time, it automatically maintains a local image cache.
When you pull an image from a remote registry (like Docker Hub or NVIDIA NGC), Apptainer saves a copy in a hidden directory. If you accidentally delete your local `.sif` file and try to pull it again, Apptainer will instantly grab it from your local cache instead of downloading gigabytes of data over a network.

To inspect how much total space you are consuming, run this command `apptainer cache list` . By default, this only shows the total file count and size. To see the specific images names, creation dates and where they were pulled from, add the verbose `-v` flag. `apptainer cache list -v`

```{note}
Under the `TYPE` column, images from Apptainer Hub appears as `shub`, while standard Docker/NGC images appear as `blob` or `library/oci`
```

Similarly, if your cluster storage quota is running low, you can clear out cached files using the `apptainer cache clean`. Running this command will ask to wipe everything. However, if you want to see what exactly would be reomved without actually deleting it, you can use `-n` or `--dry-run` flag. Moreover, to wipe out only a specific type of cached image (e.g. clearing out old Apptainer Hub templates while keeping your heavy Docker/NGC layers), you can use `apptainer cache clean --type=shub` command

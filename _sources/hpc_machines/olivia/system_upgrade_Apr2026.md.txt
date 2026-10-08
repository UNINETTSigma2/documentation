# Olivia System Upgrade -- April 2026

## System Upgrade Overview
We are pleased to announce a significant upgrade to the Olivia. This update includes enhancements designed to improve performance, scalability, and user experience. It ensures that Olivia stays up-to-date and ready for new workloads.


## Upgrades
- **Operating system:**
    - Cray Operating System (COS) upgraded to 26.3.0
- **Updated System Libraries:**
    - Libfabric to 2.3.1
    - Nvidia HPC SDK to 25.9
    - Nvidia Driver to 580.65.06
    - Default CUDA to 13.0.0
- **Backend:**
    - Upgrade of HPE Performance Cluster Manager (HPC) to 1.15
    - Upgrade of network drivers


## Software Updates
- **Updated Libraries and Tools:** 
    - Added `CUDA/13.0.0` module to `NRIS/GPU` stack
    - Updated `CrayEnv` to version 26.3
        - Added CUDA 12.9 and CUDA 13.0 modules to `CrayEnv` stack
        - Removed CUDA 11.8 and CUDA 12.6 from `CrayEnv`
- **Expanded Software Stack storage:**
    - Added 16TB more storage space to software stack to allow installation of new and additional software modules


## Important Notes
- **Action Required:**  
    - If you run code compiled using `CrayEnv` stack plus CUDA 12.6 or 11.8, you must recompile using CUDA 12.9 or 13.0
- **Action Recommended:**  
    - In case you run multi-node GPU jobs using container, update the lib fabric bind command:
        - Replace `/opt/cray/libfabric/1.22.0/lib64`  with `/opt/cray/libfabric/2.3.1/lib64`


**In case you have any new issues after the upgrade, please contact us through our [helpdesk](https://nettskjema.no/a/nris-support-request)**


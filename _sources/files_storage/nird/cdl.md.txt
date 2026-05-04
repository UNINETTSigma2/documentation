---
orphan: true
---

(nird-cdl)=

# (NIRD) Central Data Library

The NIRD Central Data Library (CDL) is a centralised data storage service built on top of the NIRD Data Lake. It serves as a shared repository for research
 projects and is designed to store data that is intended to be used and reused across multiple projects. CDL supports a wide range of data types, including 
 input datasets, shared libraries, AI models, and AI training data. By centralising commonly used data, CDL enables better organisation, efficient reuse, and 
 reduces data duplication across research projects. Data stored in the Central Data Library can be accessed through multiple protocols, including POSIX, NFS, 
 and S3. It is directly accessible from NIRD login nodes and Sigma2’s national HPC systems, and can also be seamlessly connected to third-party storage systems, 
 external computing facilities, or a user’s local desktop.

## When to use CDL service

Use the Central Data Library (CDL) when:

- You need to share and reuse data across multiple projects or research groups.
- Your data serves as common input, such as reference datasets, AI/ML training data, models, or shared libraries.
- You require centralised storage with controlled access and permissions for collaborators (access to data via the S3 protocol and S3 API.)
- Long-term storage of non-persistent input datasets enriched with metadata.

```{note}
Important: The Central Data Library is not intended for permanent archival of data, for long-term preservation you should use the {ref}`research-data-archive` instead.
```

## How to apply for CDL service

- Apply through the regular resource application process and select NIRD Data Lake as the requested storage resource.
- During the application, the Project Leader should explicitly flag the request as a Central Data Library (CDL) application.
- For more info on application acceptance process, refer to the [Central Data Library policy](https://www.sigma2.no/central-data-library-policy).

## S3 Access for the CDL

CDL uses S3-compatible object storage as the default access method for storing and retrieving data. This means you can interact with CDL data using 
standard S3 tools and clients, just like other cloud object storage services. 
For detailed, step-by-step instructions and practical examples on configuring and using S3 access, please refer to the {ref}`nird-s3` documentation.

## CDL Cache Hydrator (Caching data on HPCs)

A key functionality of the CDL service is the CDL Cache Hydrator, which extends its capabilities to HPC environments by solving a fundamental challenge in data-intensive computing: getting large input datasets from centralized storage onto the HPC system's local filesystem quickly and efficiently.
Rather than requiring users to manually stage data before each compute job, the cache hydrator automatically maintains a local read-only cache of the needed datasets on the HPC system, pulling them from NIRD via S3. Datasets can be hydrated on demand and rehydrated at any time if evicted. To keep local storage usage in check, data that has not been accessed within a configured retention period (currently 45 days) is automatically removed from the cache, while always remaining safely available on NIRD for future use.

For detailed, step-by-step instructions and practical examples of this functionality, please refer to the  {ref}`nird-cdl-cache-hydrator` documentation.










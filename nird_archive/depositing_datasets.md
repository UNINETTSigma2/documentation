---
orphan: true
---

# Depositing a dataset

The process for depositing a dataset in the RDA consists of the following stages:

- {ref}`Identify the dataset.  <Identify-the-dataset-Archive>`
- {ref}`Choose file formats. <Choose-file-formats>`
- {ref}`Log onto the web interface.<Log-onto-the-web-interface-Archive>`
- {ref}`Agree to the terms and conditions.  <Agree-to-the-terms-and-conditions-Archive>`
- {ref}`Provide metadata.  <Provide-metadata-Archive>`
- {ref}`Upload the dataset.  <Section-Upload-Dataset>`
- {ref}`Publish the dataset.  <Publish-the-dataset-Archive>`


The following subsections describe these stages.

(Identify-the-dataset-Archive)=

## Identify Dataset

Before archiving a dataset you will need to define it, make sure you have approval to archive the data and understand which type of access license should be applied to the dataset.

A dataset must be a collection of related data. Typically, this consists of a collection of files. How a dataset is arranged will vary within communities, research groups and projects. However, a guideline as to what would be accepted for archival is:

- Datasets resulting from research which is fully or in part supported by the Norwegian Research
          Council.

- Datasets of lasting value from any research discipline.

- Datasets that are not in the process of being created. Datasets should be in a state where they are
          well established (or mature) and will not be altered.

- Datasets with preferably no access restrictions so that a larger audience can make use of the data
          (i.e. it has public access). However, the Archive recognises that certain datasets of restricted
          use to a given community may be eligible for archiving.

(Choose-file-formats)=

## Choose File Formats

You should choose open file formats for your data if possible. Open file formats follow an open licence which makes it easier for people to reuse your data as it is more likely that openly available applications exist to read the data (or applications can easily be written to access the data). A list of open file formats can be found on [Wikipedia](https://en.wikipedia.org/wiki/List_of_open_file_formats). You can find more information about open data formats on the [Open Data Formats](https://opendataformats.org) site.

(Log-onto-the-web-interface-Archive)=

## Log onto the RDA Web Interface

To access the RDA web interface, direct your browser to: [https://archive.sigma2.no](https://archive.sigma2.no). You should arrive at the front page shown in Figure 1. You will need to authenticate using your FEIDE or ORCID account either by logging on via the *SIGN IN* button on the top-right or via the *ADD DATASET* button.

![rda_web_interface](imgs/figure_7_screenshot_portal_V1.png "RDA web interface")
Figure 1: Screenshot of the RDA web interface front page.

The *ADD DATASET* button provides access to the set of pages required for depositing your dataset in the RDA. These pages are accessible once you have authenticated and been allowed to access the RDA.

## Request Approval

If you have never used the RDA before you will be presented with a page informing you that you are not registered. You can submit a request approval from this page. Only approved users are allowed to deposit datasets in the RDA. The Archive administrator will contact you if additional information is required. Approval should be granted within 3 business days (and usually much sooner).

(Agree-to-the-terms-and-conditions-Archive)=

## Agree to Terms & Conditions

Once approval has been granted you will be able to deposit datasets. If you now click the *ADD DATASET* button you will be presented with a page containing a link to the Terms and Conditions as shown in Figure 2. The Terms and Conditions outline your responsibilities and those of the RDA. You will need to agree to these before you can start the deposit process.

![the_terms_and_conditions_page](imgs/figure_2_screenshot_tou.png "the terms and conditions page")
Figure 2: Screenshot of the Terms and Conditions page.

(Provide-metadata-Archive)=

## Provide Metadata

The goal of the RDA is to provide datasets that can be understood and reused. To achieve this, each dataset must contain metadata that adequately describes the dataset. The metadata page contains a list of metadata terms that will help potential users of the dataset. The mandatory terms are marked with an "*". Although the contact point and dataset owner are optional, it is good practice to provide them. Once published, the metadata fpr a dataset will be publicly accessible regardless of whether the dateset is private or not. Note, that each dataset is issued with a DOI that can be used in articles. The DOI is not publicly accessible, but will be made publicly accessible once the dataset has been published. All metadata are automatically saved (although you should check that the dataset metadata are all present and correct before publishing). Once you have completed the metadata, click the *Upload Data* button.

![rda_metadata_form](imgs/figure_3_screenshot_metadata_form.png "rda_metadata_form")
Figure 4: Screenshot of part of the RDA Metadata form.

(Section-Upload-Dataset)=

## Upload Dataset

Once the metadata has been provided you will be presented with the dataset upload page (see Figure 5). You can upload datasets from a variety of sources:

- *My Dataset* allows you to upload single files via the web interface.
- *CLI Upload* allows you to upload folders and files via the S3 protocol.
- *NIRD Project* allows you to upload files and folders that reside on the NIRD Active Storage.
- *Dropbox* allows you to upload files that are stored in Dropbox storage.

![the_upload_dataset_page](imgs/figure_4_screenshot_upload.png "upload_dataset_page")
Figure 4: Screenshot of the upload dataset page.

(Section-Upload-My-Dataset)=

### My Dataset upload

The upload occurs via the web interface, and you can upload small files (less than 5GB in size) to the archive. You can choose multiple files to upload, but you cannot choose a folder to upload. The interface allows restarts of failed uploads.

(Section-Upload-CLI)=

### Command Line Interface (CLI) upload

Choosing this mechanism for upload results in an S3 bucket to be created. You should then copy the S3 credentials and the bucket name to the configuration files to the application of your choice.

#### AWS command-line application

To use the AWS command-line application, download the application from [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), follow the instructions for configuring the client. You should now be able to upload data to the S3 container.

#### Rclone command-line application

To use the rclone application, download from [rclone](https://rclone.org) and create a configuration file (you can use the [template](./rclone-s3.conf) file and replace the "???" with the corresponding credentials). You can then upload files or folders to the S3 container. For example, `rclone --config rclone.config copy <source-file-or-folder> swift:<s3-bucket>/` which will upload the files or folders into the bucket.

(Section-NIRD-Project)=

### NIRD Project area upload

You can choose to upload a dataset that exists in the NIRD active storage or project area [https://www.sigma2.no/data-storage](https://www.sigma2.no/data-storage) to the archive storage. Once this
option is selected, you will need to log in to the login node (e.g. *ssh login.nird.sigma2.no* ). Create a manifest file in your home folder with the name `.import-archive_<dataset-identifier>` containing the paths to the files that make up the dataset. The structure of the paths should be valid arguments for the UNIX `find ! -type d` command which is used by the cron-job to get the list of files to archive.  For example if we define our dataset to consist of all gzipped tar files in the NS1234K project then the manifest file should contain the line:
/projects/NS1234K/ -name *.tar.gz

The manifest file can contain more than one line if the dataset spans more than one project or different types of files etc.

By default, the files that make up the dataset will contain the full path excluding the leading '/' (e.g. project/NS1234K/subdir1/file1.dat). You can indicate that the root part of the path be removed by adding a “//” where the root path ends.

E.g. to remove “/projects/NS2134K” from “/projects/NS1234K/subdir1/file1.dat” you would add the following to your manifest file: “/projects/NS1234K///subdir1/file1.dat”. This can be used in combination with the regular expressions and globbing that are recognised by the find command. To remove “/projects/NS1234K” from the pattern which will archive all “.tar.gz” files in the directory “/projects/NS1234K/subdir1” specify the following: “/projects/NS1234K///subdir1 -name *.tar.gz”.

The cron-job will check the home directories for a file of the form `.import-archive_<dataset-identifier>` every 15 minutes. Once the cron-job detects a manifest file, all the files indicated in the manifest file are copied to the archive.

### Dropbox upload
Choosing the dropbox option should result in a *Connect to DropBox* button that will result in a connection to dropbox being made. You can than opy files from your DropBox to the archive.

(Publish-the-dataset-Archive)=

## Publish Dataset (Archiving Data)

Once the dataset has been uploaded, and you have filled in the required metadata and any optional metadata, you can then submit your dataset for publication. If you open your dataset in `view mode` you should see a `Submit to review` button (see Figure 5).
By clicking the button, the archive manager receives the request to publish the dataset. The archive manager will review the dataset and will then approve the dataset for publication. The archive will execute a process to copy the data to the archive, create metadata files, update the archive metadata and register the DOI in DataCite. Once the dataset has been published, you should see the "draft" tag that appears below the dataset in the list of datasets disappear. The dataset will also be visible through the portal (https://archive.sigma2.no).


![the_submit_to_review_page](imgs/figure5a.png "submit_dataset_to_review")
Figure 5: Screenshot of the Submit to review button.

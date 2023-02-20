# Public Documentation for Sigma2 HPC Services

Served via [https://documentation.sigma2.no](https://documentation.sigma2.no/)

Unless explicitly noted, all text on this website is made available under the
[Creative Commons Attribution license (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
with attribution to the Sigma2/NRIS.


## How the GitHub pages are built

- Repository is mirrored from internal GitLab to GitHub using https://docs.gitlab.com/ee/user/project/repository/repository_mirroring.html#setting-up-a-push-mirror-from-gitlab-to-github.
- `gh-pages` branch is generated using https://github.com/UNINETTSigma2/documentation/blob/main/.github/workflows/sphinx.yml


## Building the documentation locally on your computer

Install dependencies as below. If these commands do not work,
try with `python3` instead of `python` (on some OS versions it has a different name):
```
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install -r requirements.txt
```

This is the nicest way to preview locally since you don't need
to re-run the command after each change:
```
$ sphinx-autobuild  . _build
```
Open `http://127.0.0.1:8000`
in your browser to view
the documentation as it should appear on the internet.

Build the HTML without opening a web server:
```
$ sphinx-build . _build
```

Build the HTML and check links:
```
$ sphinx-build -b linkcheck . _build
```


## All changes should be submitted as merge requests

In order to coordinate our efforts, changes to the documentation should be
added by _merge requests_ instead pushing directly to the `main` branch.

This allows us to review, comment, and discuss our changes before they become public.
Additionally all commits first run through the CI testing pipeline
and are only merged upon passing of all tests, reducing the chances of
accidentally breaking stuff.


## How to contribute changes

We recommend to never commit changes to the `main` branch. Consider the `main` branch
to be a read-only copy of the public documentation. Always create a new branch before changing
anything.

Assuming we want to add a section describing that no animals are allowed on the server.
We will call the branch `no-animals`.

0. Optionally but recommended: [Open an issue](https://gitlab.sigma2.no/documentation/public/issues)
   describing the necessary change. This is useful to either signal a problem if you don't have time to solve it,
   but it can also be useful to collect feedback for an idea before doing all the writing.
1. Get an up-to-date `main` branch: `git pull origin main`
2. Create a new branch for your changes. Use a short, descriptive name: `git checkout -b no-animals`
3. Edit/add files and do the changes.
4. Stage your changes: `git add new_section.md`
5. Commit the changes: `git commit -m 'Add new section'`
6. Upload your changes to the main repository: `git push origin no-animals`
7. Create a new [merge request](https://gitlab.sigma2.no/documentation/public/-/merge_requests).
   Select your branch as _source branch_ and `main` as _target branch_. Also mark the source
   branch to be deleted upon accepted merge request.
8. Describe the changes and optionally assign someone to review and approve the commits.


## How to submit a merge request via the web interface

It is possible to suggest changes and file a merge request to the documentation directly from
the [web interface](https://gitlab.sigma2.no/documentation/public):

- Browse the [file tree](https://gitlab.sigma2.no/documentation/public/-/tree/main)
  and click on the file you wish to edit.
- Click the blue button "Edit".
- Make changes to the file.
- Change the "Commit message" (bottom) to a meaningful message.
- Click "Commit changes" with "Start a new merge request with these changes" checked.


## How to update your main branch after your changes have been accepted and merged

Switch to your `main` branch:
```
$ git checkout main
```

Pull changes from `origin`
(we assume `origin` points at `git@gitlab.sigma2.no:documentation/public.git`):
```
$ git pull origin main
```


## How to review a merge request

Open merge requests are listed [here](https://gitlab.sigma2.no/documentation/public/-/merge_requests).

Every merge request is numbered, e.g. [!62](https://gitlab.sigma2.no/documentation/public/-/merge_requests/62).

Checklist before approving and merging:

- Have a look at the source branch and target branch ("Request to merge
  `source-branch` to `target-branch`).
- Click on "Changes" and scroll through the difference to see whether this
  makes sense.
- If you have questions or suggestions, you can write that under "Overview".
- You can also comment on changes directly at "Changes". You can even suggest
  changes that the submitter can accept.
- In doubt ask for clarification or involve somebody else by mentioning them
  (type "@" and name and select a suggestion).
- Once you are happy, approve.
- Check "Delete source branch".
- After that one can either "Merge" or "Merge when pipeline succeeds" (if the
  pipeline is not finished yet).
- After the merge request is merged, the changes should appear on the public
  website in a minute or two.


## List of links from sigma2.no

Before changing the paths, we need to inform the webadmin behind sigma2.no to
change them also there:

- https://documentation.sigma2.no/files_storage/nird.html (from https://www.sigma2.no/data-storage)
- https://documentation.sigma2.no/getting_help/how_to_write_good_support_requests.html (from https://www.sigma2.no/user-support)


## How to cross-reference

**Do not refer to other pages on this site by filename**. In other words, **don't do this**:
```
Linking to [some other page](../../some-page.md).
```

(the above note has been written by somebody who just fixed 60 broken internal links)

Also **do not do this** (since your local preview will then be confusing or wrong):
```
Linking to [this could be an internal link](https://documentation.sigma2.no/page/on/same/site.html).
```

Instead insert a label
and refer to the label as shown below. This is more robust since the links will
still work after the target file is moved or renamed.

Create a label at the place you want to cross-reference to:
```
(mylabel)=

## This is some section I want to point to
```

Then you can reference to it like this:
```
{ref}`mylabel`
```

Or if you want to change the link text:
```
{ref}`my text <mylabel>`
```


## Keywords and index

You can add this to your page if you want `MyKeyword` to show on the index with
a link to your page:
````
```{index} single: MyKeyword; Name of link in index
```
````

You can also have several keywords point to this page like here:
````
```{index} GPU; Name of link, OpenACC; Name of link, CUDA; Name of link
```
````

## Table of contents on longer pages

If you need a table of contents on top of a long page, do not create it manually using
anchors. Instead you can get it like this:
````
```{contents} Table of Contents
```
````

## Creating tabs

The tab plugin used is
[`sphinx-tabs`](https://sphinx-tabs.readthedocs.io/en/latest/).

To create tabs in the documentation you can use the following markdown syntax
`````
````{tabs}
```{tab} Tab 1 name

Content of tab 1
```
```{tab} Tab 2 name

Content of tab 2
```
````
`````

We can also have pure code tabs:
`````
````{tabs}
```{code-tab} c

void main() {
  printf("Hello World!\n");
}
```
```{code-tab} py

print("Hello World!")
```
````
`````


## Redirecting moved or removed pages

Sometimes we want to remove or move a page which has been linked
to from emails. In this case you can add a redirect. Look for `redirects` in `conf.py`.

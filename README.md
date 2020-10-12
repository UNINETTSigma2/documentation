# Public Documentation for Sigma2 HPC Services

Served via [https://documentation.sigma2.no](https://documentation.sigma2.no/)

Unless explicitly noted, all text on this website is made available under the
[Creative Commons Attribution license (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
with attribution to the Sigma2/Metacenter.


## How the GitHub pages are built

- Repository is mirrored from internal GitLab to GitHub using https://docs.gitlab.com/ee/user/project/repository/repository_mirroring.html#setting-up-a-push-mirror-from-gitlab-to-github.
- `gh-pages` branch is generated using https://github.com/UNINETTSigma2/documentation/blob/master/.github/workflows/sphinx.yml


## Building the documentation locally on your computer

Install dependencies:
```
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install -r requirements.txt
```

Build the HTML:
```
$ sphinx-build . _build
```

Now open `_build/index.html` in your browser. After each change
you need to `sphinx-build` again.


## All changes should be submitted as merge requests

In order to coordinate our efforts, changes to the documentation should be
added by _merge requests_ instead pushing directly to the `master` branch.

This allows us to review, comment, and discuss our changes before they become public.
Additionally all commits first run through the CI testing pipeline
and are only merged upon passing of all tests, reducing the chances of
accidentally breaking stuff.


## How to contribute changes

We recommend to never commit changes to the `master` branch. Consider the `master` branch
to be a read-only copy of the public documentation. Always create a new branch before changing
anything.

Assuming we want to add a section describing that no animals are allowed on the server.
We will call the branch `no-animals`.

0. Optionally but recommended: [Open an issue](https://scm.uninett.no/sigma2/eksterndokumentasjon/issues)
   describing the necessary change. This is useful to either signal a problem if you don't have time to solve it,
   but it can also be useful to collect feedback for an idea before doing all the writing.
1. Get an up-to-date `master` branch: `git pull origin master`
2. Create a new branch for your changes. Use a short, descriptive name: `git checkout -b no-animals`
3. Edit/add files and do the changes.
4. Stage your changes: `git add new_section.md`
5. Commit the changes: `git commit -m 'Add new section'`
6. Upload your changes to the main repository: `git push origin no-animals`
7. Create a new [merge request](https://scm.uninett.no/sigma2/eksterndokumentasjon/-/merge_requests).
   Select your branch as _source branch_ and `master` as _target branch_. Also mark the source
   branch to be deleted upon accepted merge request.
8. Describe the changes and optionally assign someone to review and approve the commits.


## How to submit a merge request via the web interface

It is possible to suggest changes and file a merge request to the documentation directly from
the [web interface](https://scm.uninett.no/sigma2/eksterndokumentasjon):

- Browse the [file tree](https://scm.uninett.no/sigma2/eksterndokumentasjon/-/tree/master)
  and click on the file you wish to edit.
- Click the blue button "Edit".
- Make changes to the file.
- Change the "Commit message" (bottom) to a meaningful message.
- Click "Commit changes" with "Start a new merge request with these changes" checked.


## How to update your master branch after your changes have been accepted and merged

Switch to your `master` branch:
```
$ git checkout master
```

Pull changes from `origin`
(we assume `origin` points at `git@scm.uninett.no:sigma2/eksterndokumentasjon.git`):
```
$ git pull origin master
```


## How to review a merge request

Open merge requests are listed [here](https://scm.uninett.no/sigma2/eksterndokumentasjon/-/merge_requests).

Every merge request is numbered, e.g. [!62](https://scm.uninett.no/sigma2/eksterndokumentasjon/-/merge_requests/62).

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



# Storage performance: what to avoid

- Avoid having a **large number of files in a single directory** and
  rather split files in multiple sub-directories.
- **Avoid repetitive `stat`** operations because it creates a significant
  load on the file system.
- **Do not use `ls -l`** on large directories, because it is slow.  Rather
  use `ls` and run `ls -l` only for the specific files you need
  extended information about.

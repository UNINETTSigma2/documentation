# Job Array Howto

In this example we wish to run many similar sequential jobs in
parallel using job arrays. We take Python as an example but this does
not matter for the job arrays:

```{eval-rst}
.. literalinclude:: files/array_test.py
  :language: python
```

Download the script:
```{eval-rst}
:download:`files/array_test.py`
```

Try it out:

```bash
$ python array_test.py
start at 15:23:48
sleep for 10 seconds ...
stop at 15:23:58
```

Good.  Now we would like to run this script 16 times at (more or less) the same
time.  For this we use the following

```{eval-rst}
.. literalinclude:: files/array_howto.sh
  :language: bash
```

Download the script:
```{eval-rst}
:download:`files/array_howto.sh`
```

This is a script for running a _normal_ array job on Saga.  It can
easily be changed to run on Fram or use a different job type.

Submit the script with `sbatch` and after a while you should see 16
output files in your submit directory:

```bash
$ ls -l output*txt
-rw------- 1 user user 60 Oct 14 14:44 output_1.txt
-rw------- 1 user user 60 Oct 14 14:44 output_10.txt
-rw------- 1 user user 60 Oct 14 14:44 output_11.txt
-rw------- 1 user user 60 Oct 14 14:44 output_12.txt
-rw------- 1 user user 60 Oct 14 14:44 output_13.txt
-rw------- 1 user user 60 Oct 14 14:44 output_14.txt
-rw------- 1 user user 60 Oct 14 14:44 output_15.txt
-rw------- 1 user user 60 Oct 14 14:44 output_16.txt
-rw------- 1 user user 60 Oct 14 14:44 output_2.txt
-rw------- 1 user user 60 Oct 14 14:44 output_3.txt
-rw------- 1 user user 60 Oct 14 14:44 output_4.txt
-rw------- 1 user user 60 Oct 14 14:44 output_5.txt
-rw------- 1 user user 60 Oct 14 14:44 output_6.txt
-rw------- 1 user user 60 Oct 14 14:44 output_7.txt
-rw------- 1 user user 60 Oct 14 14:44 output_8.txt
-rw------- 1 user user 60 Oct 14 14:44 output_9.txt
```

Observe that they all started (approximately) at the same time:

```bash
$ grep start output*txt
output_1.txt:start at 14:43:58
output_10.txt:start at 14:44:00
output_11.txt:start at 14:43:59
output_12.txt:start at 14:43:59
output_13.txt:start at 14:44:00
output_14.txt:start at 14:43:59
output_15.txt:start at 14:43:59
output_16.txt:start at 14:43:59
output_2.txt:start at 14:44:00
output_3.txt:start at 14:43:59
output_4.txt:start at 14:43:59
output_5.txt:start at 14:43:58
output_6.txt:start at 14:43:59
output_7.txt:start at 14:43:58
output_8.txt:start at 14:44:00
output_9.txt:start at 14:43:59
```

# Queue System Concepts

__FIXME: This is currently only a dumping ground for what might be in
here.__



Jobs are administered and allocated resources based on the following:
1. **Partitions** - logical groupings of nodes that can be thought of as queues.  `normal`, `bigmem`, `optimist` partitions.
2. **QoS** (Quality of Service). QoSes are administered to `normal` and `bigmem`
	partitions. Special QoSes are `devel`, `preproc`.
3. **Account** "nn9999k" per project, with CPU hour quota. "nn9999x" are *optimist* jobs, with a **QoS** `optimist`
and separate CPU hour quotas.

Perhaps: Jobs have job steps executed after each other (usually), and
each step can run one or more tasks in parallel.

For an overview of the Slurm concepts, Slurm has an excellent beginners guide: [Quick Start](https://slurm.schedmd.com/quickstart.html).

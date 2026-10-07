---
orphan: true
---

# Distributed LLM Fine-Tuning & Inference on HPC systems, Fall 2026

(2026-LLM-Course-Bergen-v2)=

NRIS Training is organizing a third round of Distributed LLM Fine-Tuning & Inference on HPC systems. This is a two-day, in-person, hands-on course in Bergen. Gain practical, hands-on experience working with single-GPU fine-tuning, multi-GPU scaling on single- and multi-node setups, and optimized LLM inference on a high-performance computing (HPC) system. Attend this course to build applied skills in optimizing large language models in HPC environments.

**When:** November 18.-19., 2026

**Where:** Bergen, University Campus

**Instructor:** [Hicham Agueny](https://www.linkedin.com/in/hicham-agueny-956a1368/)

**HPC System:** [Olivia](https://www.sigma2.no/meet-olivia-norways-next-supercomputer)


<details>
<summary><h2 style="display: inline;">Course program and schedule</h2></summary>

<H2> Day 1 — Single-GPU Fine-Tuning & HPC Foundations

**Theme:** Build an efficient single-GPU fine-tuning workflow on an HPC system.

<H3> Morning Session (09:30–12:00) — HPC Fundamentals & Fine-Tuning Optimization

1. **HPC Foundations for LLM Workloads**
   - Overview of Olivia Supercomputer
   - Containerized environments including EESSI
2. **LLM Fine-Tuning Fundamentals**
   - Parameter-efficient fine-tuning with LoRA
   - Quantized fine-tuning with QLoRA

<H3> Afternoon Session (13:00–15:30) — Hands-On: Single-GPU workflow for QA and XSum Tasks

- LoRA fine-tuning workflow
- Quantized fine-tuning with QLoRA: FP4 vs BF16 comparison
- Evaluation of the fine-tuned model
- GPU monitoring and memory profiling

<H3> Wrap-Up & Discussion (15:30–16:00)

**Outcome:** Participants implement and optimize a complete single-GPU fine-tuning pipeline with performance diagnostics on an HPC system.

<H2> Day 2 — Distributed Training & Optimized Inference

**Theme:** Scale fine-tuning and inference across multiple GPUs while minimizing communication overhead.

<H3> Morning Session (09:30–12:00) — Distributed Fine-Tuning

1. **Distributed Training Concepts**
   - Concept of parallelism
   - DDP vs FSDP
   - Communication and scaling efficiency
2. **Hands-On: Multi-GPU Fine-Tuning on a single node & acorss nodes for QA and XSum Tasks**
   - Multi-GPU & multi-node LoRA & QLoRA fine-tuning
   - Evaluation of the fine-tuned model accros multi-GPUs
   - Profiling distributed workloads

<H3> Afternoon Session (13:00–15:30) — Hands-On: Optimized Inference

- Introduction to the vLLM inference engine
- Single-GPU inference benchmarking
- Quantization: torchao, bitsandbytes, GPTQModel
- Multi-GPU inference

<H3> Wrap-Up & Discussion (15:30–16:00)

**Outcome:** Participants scale fine-tuned models and inference across multiple GPUs, interpret performance metrics, and apply optimization strategies suitable for HPC allocations.

</details>
<br>

## Target audience & prerequisites
The course is ideal for researchers, developers, and students with Python experience who want hands-on skills in scalable LLM training and inference on an HPC system.

**Registration:** [Register here](https://docs.google.com/forms/d/e/1FAIpQLSe9so1ZdO4_0DEYmf7MLhBvfHkCh9RGNWlZd2cM-Co3m3lriA/viewform?usp=dialog)

## Practical Information
The course is free of charge, but will have a maximum capacity of 25 people. 
Lunch will be included, and coffee/tea will be served.

## Contact us
If there are questions regarding the course or NRIS Training, please contact us at **training@nris.no**.
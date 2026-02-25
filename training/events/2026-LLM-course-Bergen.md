# Distributed LLM Fine-Tuning & Inference on HPC Systems: March 25-26 2026

The Norwegian Research Infrastructure Services (NRIS) is hosting a two-day in-person, hands-on physical course in Bergen. Gain practical, hands-on experience over two days working with single-GPU fine-tuning, multi-GPU scaling, and optimized LLM inference on a high-performance computing (HPC) system. Build applied skills in optimizing large language models in HPC environments.

**When:** 25th and 26th of March, 2026, 09:30-16:00 both days

**Where:** [Nygårdsgaten 5, Bergen](https://www.google.com/maps/place/Studieadministrativ+avd.,+Strandgaten+10,+5013+Bergen/@60.3889945,5.3244845,51m/data=!3m1!1e3!4m6!3m5!1s0x463cfea98bbfbf9f:0xb05a6eaa50556379!8m2!3d60.3889736!4d5.3246939!16s%2Fg%2F11q2vvqr9d?entry=ttu&g_ep=EgoyMDI2MDIyMi4wIKXMDSoASAFQAw%3D%3D)
(exact conference room will be provided)

**Instructor**: [Hicham Agueny](https://www.linkedin.com/in/hicham-agueny-956a1368/)

**Content:** In this course, you will learn to:
* Implement parameter-efficient fine-tuning using LoRA and QLoRA
* Configure and launch distributed training workloads across multiple GPUs
* Perform distributed LLM inference
* Monitor and analyze GPU utilization and profiling GPU memory

<details>
  <summary>More information and course schedule</summary>
    <br>

## Day 1 — Single-GPU Fine-Tuning & HPC Foundations

**Theme:** Build an efficient single-GPU fine-tuning workflow on an HPC system.

**Morning Session (09:30–12:00) — HPC Fundamentals & Fine-Tuning Optimization**
1. **HPC Foundations for LLM Workloads**
    * Overview of Olivia Supercomputer
    * Storage hierarchy strategy
    * Containerized environments

2. **LLM Fine-Tuning Fundamentals**
    * Parameter-efficient fine-tuning (LoRA, QLoRA)
    * Quantization within QLoRA (FP4, FP8, BF16)
    * Memory–throughput trade-offs

**Afternoon Session (13:00–15:30) — Hands-On: Single-GPU Workflow**
* End-to-end LoRA fine-tuning workflow
* Quantized fine-tuning: FP4 vs FP8 vs BF16 comparison
* GPU monitoring and memory profiling

**Wrap-Up & Discussion (15:30–16:00)**

**Outcome:** Participants implement and optimize a complete single-GPU fine-tuning pipeline with performance diagnostics on an HPC system.

## Day 2 — Distributed Training & Optimized Inference

**Theme:** Scale fine-tuning and inference across multiple GPUs while minimizing communication overhead.

**Morning Session (09:30–12:00) — Distributed Fine-Tuning**

1. **Distributed Training Concepts**
    * DDP vs FSDP
    * Communication overhead and scaling efficiency

2. **Hands-On: Multi-GPU Fine-Tuning**
    * Multi-GPU LoRA & QLoRA fine-tuning
    * Profiling distributed workloads
    * Throughput and scaling efficiency analysis

**Afternoon Session (13:00–15:30) — Hands-On: Optimized Inference**

* Introduction to the vLLM inference engine
* Single-GPU inference benchmarking
* Multi-GPU inference scaling
* Latency vs throughput trade-offs

**Wrap-Up & Discussion (15:30–16:00)**

**Outcome:** Participants scale fine-tuned models and inference across multiple GPUs, interpret performance metrics, and apply optimization strategies suitable for HPC allocations.

--- 

</details>
<br>

**Target audience:** This course is ideal for researchers, developers, and students with Python experience who want hands-on skills in scalable LLM training and inference on an HPC system. 

**Prerequisites:**
- Familiarity with machine learning (ML) frameworks (e.g., PyTorch)
- Basic understanding of large language models (LLMs)

**Registration:** To be announced.

**Practical information**: The course is free of charge, and has a maximum capacity of 30 participants. There will be serving of food, some light pastries and coffee/tea both days.

**Contact person**: NRIS Training Coordinator Eirik Skjerve (<eirik.skjerve@nris.no>)
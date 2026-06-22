# High-Performance Concurrency & Workload Optimization

[![Language](https://img.shields.io/badge/Language-C%2B%2B-blue.svg)](https://isocpp.org/)
[![Frameworks](https://img.shields.io/badge/Frameworks-CUDA%20%7C%20OpenCL%20%7C%20OpenMP%20%7C%20Pthreads-orange.svg)]()

A high-performance computing (HPC) and advanced concurrency sandbox dedicated to optimizing workloads across heterogeneous hardware architectures. This repository features two core engineering tracks: deterministic **Numerical Integration Solvers** optimized across mass-parallel GPU and CPU layers, and a production-grade, asynchronous **Multi-Threaded Pipeline Simulation** modeling strict resource contention.

The core objective is to isolate, profile, and mitigate hardware-specific bottlenecks—such as synchronization overhead, thread contention, memory latency, and race conditions—under high-throughput constraints.

---

## 📂 Repository Architecture & Subsystems

```text
Parallel-Programming/ (Root)
│
├── 📂 Workload-Distribution-using-CUDA-OpenCL/  ──► [Track 1: GPU Hardware Acceleration]
│   ├── code
│   └── cuda-opencl-report.pdf
│
├── 📂 Workload-Distribution-using-OpenMP/       ──► [Track 1: OpenMP]
│   ├── code
│   └── OpenMP-report.pdf
│
├── 📂 Workload-Distribution-using-pthreads/     ──► [Track 1: Low-Level OS Threading & Locks]
│   ├── code
│   └── pthread-report.pdf
│
└── 📂 multithread-pizza-store-simulation/       ──► [Track 2: Asynchronous Event-Driven Pipeline]
    └── code
⚙️ Track 1: Computational Workload Distribution (Numerical Solvers)
These modules benchmark an identical mathematical workload across different abstraction layers to analyze hardware exploitation efficiency. Detailed performance profiles are compiled within the embedded PDF reports inside each subdirectory.

Massively Parallel Heterogeneous Computing (GPU) (Workload-Distribution-using-CUDA-OpenCL)

Comparative evaluation of native NVIDIA CUDA against cross-platform OpenCL.

Optimization of Streaming Multiprocessor (SM) occupancy and warp execution efficiency via fine-tuning threads-per-block metrics.

Elimination of global memory bus saturation by implementing explicit Shared Memory/Local Scratchpad caching.

Profiling hardware barrier overhead (__syncthreads()) during parallel data reduction.

Multi-Core Shared-Memory Parallelism (CPU) (Workload-Distribution-using-OpenMP)

Evaluation of loop-based vs. deep recursive task-based scheduling.

Load balancing via data-reduction primitives to eliminate load imbalance across asymmetric CPU cores.

Native OS Kernel Threading & Core Synchronization (Workload-Distribution-using-pthreads)

Direct POSIX Threads (Pthreads) lifecycle management (manual thread-spawning, load-decomposition boundaries, and join-fences).

Comprehensive lock contention profiling: evaluating Mutex synchronization blocks (integration-using-locks.cpp) against completely lock-free vectorized architectures (integration-no-locks.cpp).

🍕 Track 2: Asynchronous Event-Driven Architecture (Systems Application)
Multi-Threaded Pipeline Simulation (multithread-pizza-store-simulation)

Architecture: A real-world application of the Producer-Consumer design pattern, modeling an asynchronous, multi-stage transaction pipeline (Order Placement ➔ Preparation ➔ Baking ➔ Delivery).

Concurrency Controls: Implements POSIX threads managed entirely via native synchronization primitives. Uses explicit Mutex Locks to protect shared states (available ingredients, oven slots, delivery drivers) and Condition Variables to handle non-blocking thread signaling and context switching.

Failure Mitigation: Engineered to prevent catastrophic multi-threading failures such as deadlocks, thread starvation, and race conditions under continuous, high-volume asynchronous order traffic.

Artifacts: Includes execution metrics, simulation configurations, and architectural analysis in the pthread-report.pdf.

📈 High-Performance Engineering Key Insights
Thread Contention vs. Throughput: Profiling the numerical solvers verified that native mutex constraints degrade performance exponentially under heavy core scaling. Vectorized reduction and lock-free execution pathing achieved near-linear scaling metrics.

Pipeline Synchronization Efficiency: In the asynchronous storefront simulation, fine-grained locking over distinct resource groups (e.g., separating oven blocks from driver pools) drastically reduced thread blocking states compared to a coarse global lock approach.

Compute vs. Memory Bounds: GPU kernels are bound heavily by memory access coalescence and hardware barrier latency, while deep-recursive CPU tasks are latency-bound by stack allocation and OS task-scheduling metadata overhead.

🛠️ Toolchain & Compilation
Ensure your target system has the necessary compiler infrastructures and hardware-vendor backends configured.

Prerequisites
GPU Targets: NVIDIA CUDA Toolkit (v11.0+) or an OpenCL 2.0+ compatible runtime ICD.

CPU Targets: GCC Core toolchain (v9.0+) with native OpenMP (-fopenmp) and POSIX standard threading (-pthread) support.

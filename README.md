# High-Performance Numerical Integration Solvers

[![Language](https://img.shields.io/badge/Language-C%2B%2B-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Frameworks](https://img.shields.io/badge/Frameworks-CUDA%20%7C%20OpenCL%20%7C%20OpenMP%20%7C%20Pthreads-orange.svg)]()

A high-performance computing (HPC) sandbox dedicated to optimizing mathematical workloads across heterogeneous hardware architectures. This repository features a comparative performance analysis of deterministic **Numerical Integration Solvers**, engineered across multiple abstraction layers spanning mass-parallel GPU computing and low-level CPU concurrency.

The core engineering objective is to isolate, profile, and mitigate hardware-specific bottlenecks—such as memory latency, synchronization overhead, thread contention, and cache locality—under an identical mathematical workload.

---

## System Architecture & Paradigm Breakdown

The infrastructure evaluates workload distribution paradigms across three distinct native computing tiers. Detailed technical analysis and benchmarks are compiled within the embedded PDF reports inside each subdirectory.

### 1. Massively Parallel Heterogeneous Computing (GPU)
`Workload-Distribution-using-CUDA-OpenCL`
* **Architectures:** NVIDIA CUDA (Native) vs. OpenCL (Cross-platform framework).
* **Hardware Benchmarks:** Fine-tuning threads-per-block metrics to optimize Streaming Multiprocessor (SM) occupancy and warp execution efficiency.
* **Memory Subsystem Tuning:** Comparative evaluation of Global Memory bandwidth limits against explicit **Shared Memory/Local Scratchpad** caching to eliminate global memory bus saturation.
* **Synchronization Barriers:** Profiling hardware barrier overhead (`__syncthreads()`) and work-group fence primitives during parallel data reduction.

### 2. Multi-Core Shared-Memory Parallelism (CPU)
`Workload-Distribution-using-OpenMP`
* **Execution Paradigms:** Comparative routing evaluating traditional loop-based parallelism against explicit multi-threaded Task-Based scheduling.
* **Load Balancing:** Analysis of recursive vs. non-recursive task-splitting under strict data-reduction primitives to eliminate load imbalance across asymmetric CPU cores.

### 3. Native OS Kernel Threading & Synchronization (Low-Level)
`Workload-Distribution-using-pthreads`
* **Concurrency Control:** Hard-coded POSIX Threads (Pthreads) management implementing explicit thread-spawning, load-decomposition boundaries, and join-fences.
* **Lock Contention Profiling:** Comprehensive algorithmic trade-off evaluation comparing:
  * Strict synchronization wrappers using native Mutex blocks (`integration-using-locks.cpp`).
  * Lock-free vectorized architectures to completely eliminate context-switching overheads (`integration-no-locks.cpp`).
  * Intermittent work distribution utilizing non-sequential jump patterns (`integration_with_jumps.cpp`).

---

## High-Performance Engineering Key Insights

* **Thread Contention vs. Throughput:** Profiling verified that native mutex constraints degrade performance exponentially under heavy core scaling due to lock-acquisition latency. Vectorized reduction and lock-free execution pathing achieved near-linear scaling metrics.
* **Compute vs. Memory Bounds:** GPU kernels are bound heavily by memory access coalescence and hardware-level synchronization overhead, while deep-recursive CPU tasks are latency-bound by stack allocation and task-scheduling metadata overhead.
* **Hardware Exploitation:** Isolating performance profiles under identical mathematical workloads provides precise metrics regarding execution efficiency, enabling optimal framework selection based on hardware capabilities (embedded vs. host environments).

---

## Toolchain & Compilation

To compile and profile the kernels locally, ensure your target system has the necessary compiler infrastructures and hardware-vendor backends configured.

### Prerequisites
* **GPU Targets:** NVIDIA CUDA Toolkit (v11.0+) or an OpenCL 2.0+ compatible runtime ICD.
* **CPU Targets:** GCC Core toolchain (v9.0+) with native OpenMP (`-fopenmp`) and POSIX standard threading (`-pthread`) support.

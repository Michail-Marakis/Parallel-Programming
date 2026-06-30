# Systems Programming & Hardware

[![Language](https://img.shields.io/badge/Language-C%20%7C%20C%2B%2B%20%7C%20VHDL-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)](https://github.com/Michail-Marakis/Parallel-Programming)
[![Technologies](https://img.shields.io/badge/Technologies-CUDA%20%7C%20OpenCL%20%7C%20OpenMP%20%7C%20Pthreads%20%7C%20VHDL-orange.svg)]()

A collection of systems programming and hardware design projects focused on parallel computing, concurrency, and low-level software development. The repository explores different workload distribution strategies across CPUs and GPUs, synchronization techniques, multithreaded applications, and includes a simple digital hardware design project implemented in VHDL.

The primary objective is to study performance, scalability, synchronization, memory behavior, and hardware-aware programming through practical implementations.

---

## Repository Structure

> **Systems-Programming-Hardware/** *(Root)*
>
> * **Workload-Distribution-using-CUDA-OpenCL/** ──► *GPU Parallel Computing*
>     * `CUDA_barrier_experiment.cu`
>     * `CUDA_diffrent_thread_per_block.cu`
>     * `CUDA_diffrent_workload_distribution.cu`
>     * `CUDA_memory_models.cu`
>     * `OPENCL_barrier_experimen.c`
>     * `OPENCL_diffrent_thread_per_block.c`
>     * `OPENCL_diffrent_workload_distribution_CPU.c`
>     * `OPENCL_memory_models.c`
>     * `cuda-opencl-report.pdf`
>
> * **Workload-Distribution-using-OpenMP/** ──► *Shared-Memory Parallelism*
>     * `integration_using_loop_with_reduction.cpp`
>     * `integration_using_loop_with_no_reduction.cpp`
>     * `integration_task_based_recursion.cpp`
>     * `integration_task_based_no_recursion.cpp`
>     * `integration_alternative_routing.cpp`
>     * `OpenMP-report.pdf`
>
> * **Workload-Distribution-using-pthreads/** ──► *POSIX Threads*
>     * `integration.cpp`
>     * `integration_using_locks.cpp`
>     * `integration-no-locks.cpp`
>     * `integration-dynamic-sharing-job-queue.cpp`
>     * `integration_with_jumps.cpp`
>     * `pthread-report.pdf`
>
> * **multithread-pizza-store-simulation/** ──► *Multithreaded Pipeline Simulation*
>     * `thread-handling-pizza.c`
>     * `thread-handling-pizza.h`
>     * `test-res.sh`
>
> * **16-bit-ALU-VHDL/** ──► *Introductory Digital Hardware Design*
>     * alu_system.vhd
>     * alu_components.vhd

---

## Track 1: Computational Workload Distribution (Numerical Solvers)

These modules benchmark an identical numerical integration workload across different abstraction layers to analyze hardware exploitation efficiency and programming model trade-offs.

### Heterogeneous GPU Computing (`Workload-Distribution-using-CUDA-OpenCL`)
* **Hardware Barrier Synchronization:** Profiling execution overhead and work-group fence efficiency during parallel reductions (`CUDA_barrier_experiment.cu`, `OPENCL_barrier_experimen.c`).
* **Occupancy Tuning:** Thread-per-block configuration profiling to determine optimal streaming multiprocessor wrapping (`CUDA_diffrent_thread_per_block.cu`, `OPENCL_diffrent_thread_per_block.c`).
* **Workload Mapping:** Comparative analysis of alternative spatial data distribution patterns over the GPU grid vs. host CPU routing (`CUDA_diffrent_workload_distribution.cu`, `OPENCL_diffrent_workload_distribution_CPU.c`).
* **Memory Subsystem Layouts:** Evaluation of explicit Shared Memory / Local Scratchpad architectures against direct Global Memory bus access to eliminate bandwidth saturation (`CUDA_memory_models.cu`, `OPENCL_memory_models.c`).

### Shared-Memory Compiler Parallelism (`Workload-Distribution-using-OpenMP`)
* **Loop-Based Parallelization:** Testing traditional loop constructs with automated variable privatization, comparing built-in parallel reductions against raw un-synchronized loops (`integration_using_loop_with_reduction.cpp`, `integration_using_loop_with_no_reduction.cpp`).
* **Task-Based Concurrency:** Implementing dynamic compiler tasks to evaluate recursive divide-and-conquer processing against flat non-recursive queue routing (`integration_task_based_recursion.cpp`, `integration_task_based_no_recursion.cpp`).
* **Custom Scheduling Paths:** Engineering alternative work-routing algorithms to minimize scheduling overhead across asymmetric CPU threads (`integration_alternative_routing.cpp`).

### Native Low-Level OS Threading (`Workload-Distribution-using-pthreads`)
* **Baseline Concurrency:** Raw POSIX thread spawning, manual domain decomposition, and join fencing boundaries (`integration.cpp`).
* **Lock Contention Analysis:** Profiling the severe performance penalties of heavy thread contention using native Mutex locks against fully optimized lock-free architectures (`integration_using_locks.cpp`, `integration-no-locks.cpp`).
* **Dynamic Work Distribution:** Implementing an asynchronous dynamic sharing Job Queue paradigm to eliminate load imbalance among operating system threads (`integration-dynamic-sharing-job-queue.cpp`).
* **Non-Sequential Traversal:** Measuring cash locality and performance profiles using interleaved work distribution via custom jump step patterns (`integration_with_jumps.cpp`).

---

## Track 2: Asynchronous Event-Driven Architecture (Systems Application)

### Multi-Threaded Pipeline Simulation (`multithread-pizza-store-simulation`)
* **Architecture:** A complete multi-stage, event-driven transaction engine (`thread-handling-pizza.c`, `thread-handling-pizza.h`) modeling a high-throughput storefront via the **Producer-Consumer design pattern**.
* **Thread Lifecycle Pipelines:** Orders move asynchronously through decoupled operational stages managed entirely via POSIX threads (Order Intake ➔ Kitchen Prep ➔ Oven Baking ➔ Logistics/Delivery).
* **Granular Synchronization:** Protecting global inventory, limited oven slots, and delivery drivers using fine-grained **Mutex Locks** and handling non-blocking thread communication via **Condition Variables**.
* **Race & Deadlock Mitigation:** Hardened architecture engineered to guarantee fail-safe state changes under heavy concurrent stress, eliminating race conditions, thread starvation, and circular wait deadlocks.
* **Automated Validation:** Includes a functional test harness (`test-res.sh`) to stress-test the pipeline and extract execution telemetry under varying load profiles.

---

## High-Performance Engineering Key Insights

* **Thread Contention vs. Throughput:** Profiling the numerical solvers verified that native mutex constraints degrade performance exponentially under heavy core scaling. Vectorized reduction and lock-free execution pathing achieved near-linear scaling metrics.
* **Pipeline Synchronization Efficiency:** In the asynchronous storefront simulation, fine-grained locking over distinct resource groups drastically reduced thread blocking states compared to a coarse global lock approach.
* **Compute vs. Memory Bounds:** GPU kernels are bound heavily by memory access coalescence and hardware barrier latency, while deep-recursive CPU tasks are latency-bound by stack allocation and OS task-scheduling metadata overhead.

---

## Toolchain & Compilation

Ensure your target system has the necessary compiler infrastructures and hardware-vendor backends configured.

### Prerequisites
* **GPU Targets:** NVIDIA CUDA Toolkit (v11.0+) or an OpenCL 2.0+ compatible runtime ICD.
* **CPU Targets:** GCC Core toolchain (v9.0+) with native OpenMP (`-fopenmp`) and POSIX standard threading (`-pthread`) support.

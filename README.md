# Systems Programming & Hardware

[![Language](https://img.shields.io/badge/Language-C%20%7C%20C%2B%2B%20%7C%20VHDL-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)](https://github.com/Michail-Marakis/Systems-Programming-Hardware)
[![Technologies](https://img.shields.io/badge/Technologies-CUDA%20%7C%20OpenCL%20%7C%20OpenMP%20%7C%20Pthreads%20%7C%20VHDL-orange.svg)]()

The projects were developed as part of university coursework and personal experimentation. The repository contains implementations using CUDA, OpenCL, OpenMP, POSIX Threads, and VHDL, focusing on workload distribution, synchronization, multithreaded systems, and basic hardware architecture.

The projects were developed as part of university coursework and personal experimentation to better understand performance, scalability, synchronization mechanisms, and hardware-oriented programming.

---

## Repository Structure

> **Systems-Programming-Hardware/**
>
> * **Parallel-Programming/**
>     * **Workload-Distribution-using-CUDA-OpenCL/** ──► *GPU Computing*
>     * **Workload-Distribution-using-OpenMP/** ──► *Shared-Memory Parallelism*
>     * **Workload-Distribution-using-pthreads/** ──► *POSIX Threads*
>     * **multithread-pizza-store-simulation/** ──► *Concurrent Systems Simulation*
>
> * **VHDL/**
>     * **16-bit-ALU/** ──► *Introductory Digital Hardware Design*

---

# Parallel Programming

## CUDA & OpenCL

Different implementations of the same numerical integration problem were developed using CUDA and OpenCL to compare workload distribution strategies across heterogeneous hardware.

Experiments include:

- Barrier synchronization
- Thread-per-block configuration
- Workload distribution
- GPU memory models (Global vs Shared/Local Memory)

---

## OpenMP

Numerical integration implementations using OpenMP with different parallelization techniques:

- Loop parallelization
- Reduction operations
- Task-based recursion
- Dynamic scheduling strategies

---

## POSIX Threads (pthreads)

Manual multithreaded implementations focusing on synchronization and workload balancing.

Topics explored:

- Thread creation and synchronization
- Mutex-based locking
- Lock-free execution
- Dynamic job queues
- Alternative work distribution strategies

---

## Multi-threaded Pizza Store Simulation

A producer-consumer simulation of a pizza delivery system implemented with POSIX Threads.

Features include:

- Multiple concurrent worker threads
- Mutexes and Condition Variables
- Shared resource synchronization
- Order processing pipeline
- Automated testing script

---

# VHDL

## Introductory Hardware Design (VHDL)

A simple VHDL project implementing a modular 16-bit Arithmetic Logic Unit (ALU). The design includes a Full Adder, a reusable 1-bit ALU slice, opcode-based control logic, and a scalable 16-bit ALU built using structural design principles.

### Components
- Full Adder
- 1-bit ALU Slice
- 16-bit ALU
- Opcode Control Unit

> **Roadmap:** Additional VHDL projects covering topics from my Computer Architecture coursework will be added as the course progresses.

## Toolchain

### CUDA / OpenCL

- NVIDIA CUDA Toolkit (11+)
- OpenCL 2.0 compatible runtime

### CPU

- GCC (9+)
- OpenMP
- POSIX Threads

### Hardware Design

- Quartus Prime / ModelSim (or any VHDL-compatible simulator)
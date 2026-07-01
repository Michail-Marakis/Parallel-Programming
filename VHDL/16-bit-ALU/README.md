# 16-bit ALU Design in VHDL

This folder contains a simple VHDL implementation of a 16-bit Arithmetic Logic Unit (ALU).  
The project is divided into two parts: a 1-bit ALU slice with a Full Adder, and a 16-bit ALU constructed by cascading sixteen 1-bit slices. The implementation supports arithmetic and basic logic operations and was verified through simulation.

---

## alu_components.vhd

Implements the basic hardware building blocks:
- Full Adder
- 1-bit ALU Slice (AND, OR, XOR, ADD)

### RTL Diagram
<img src="https://github.com/user-attachments/assets/58e77b3c-14bb-43ee-968e-2caf4e9bea1c" width="700"/>

### Addition
<img src="https://github.com/user-attachments/assets/f57e989b-5287-4601-91fe-f78f7e5861e0" width="700"/>

### Logic Operations
<img src="https://github.com/user-attachments/assets/dc2b39bb-ec7d-4d1e-bc91-6e101ca987df" width="700"/>

---

## alu_system.vhd

Builds a complete 16-bit ALU using sixteen 1-bit ALU slices connected through carry propagation. A control unit decodes the opcode and selects the required arithmetic or logic operation.

### RTL Diagram
<img src="https://github.com/user-attachments/assets/62409a71-2241-49c9-8bb3-1d04d299144e" width="700"/>

### 16-bit Addition
<img src="https://github.com/user-attachments/assets/d6f3ece8-56d2-4220-9195-662527460f4f" width="700"/>

### 16-bit Subtraction
<img src="https://github.com/user-attachments/assets/1bff6188-9dcd-4648-95e0-fb3f9b9fe4a3" width="700"/>

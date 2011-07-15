2D Heisenberg Model on GPU
==========================

Author: Katharine Hyatt (kshyatt@uwaterloo.ca)
Date: July 15 2011

Introduction
-------------------
This code is intended to simulate the 2D Heisenberg model on a 4 x 4 square lattice. 
It generates the sparse Hamiltonian and then applies the Lanczos method to it. 

The GPU code (testhamiltonian) is based off of the CPU code (GenHam) written by Roger Melko. 

I want to extend the Hamiltonian generating code to work with different lattice sizes and different models (XY, etc).

What Works and What Doesn't
-------------------------------------
The makefile is used to compile the CPU code. All the CPU code (ED_Lan, Lanczos_07, GenHam, lapack, param) works. 

The CUDA files testhamiltonian.cu and testhamiltonian.h can be compiled and run.
lanczos.cu will compile but will probably not run.

To compile testhamiltonian.cu:

nvcc -arch=sm_20 testhamiltonian.cu




2D Heisenberg Model on GPU
==========================

Author: Katharine Hyatt (kshyatt@uwaterloo.ca)
Date: July 15 2011

Introduction
-------------------
This code is intended to simulate the 2D Heisenberg model on a 4 x 4 square lattice. 
It generates the sparse Hamiltonian in COO format and then applies the Lanczos method to it. 

The GPU code (testhamiltonian) is based off of CPU code written by Roger Melko. 

What Works and What Doesn't
-------------------------------------
The makefile will compile all the code. See nVidia's nvcc guide for more compiler flags you can pass. 

Right now testhamiltonian and lanczos should work to deliver the correct eigenvalues.

To Do
--------------------------------------
* Add more model possibilities to testhamiltonian
* Improve the sorting speeds on testhamiltonian
* Change the way lanczos looks for convergence
* Improve the diagonalization scheme in lanczos
* Add benchmarking information
* Add stream functionality (allow multiple Hamiltonians to be constructed at once, multiple Lanczos methods running simulaneously)

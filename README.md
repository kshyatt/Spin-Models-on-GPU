2D Heisenberg Model on GPU
==========================

Author: Katharine Hyatt (kshyatt AT uwaterloo DOT ca)
Date: February 7 2012

Introduction
-------------------
This code is intended to simulate the 2D Heisenberg model on a variety of square lattices. 
It generates the sparse Hamiltonian in COO format and then applies the Lanczos method to it. 

The GPU code is based off of CPU code written by Roger Melko. 

Requirements
------------------
Since this code is CUDA based, you will need an nVidia GPU as well as the latest version of the CUDA toolkit and developer driver. The Hamiltonian-generating part also relies on Sean Baxter's Modern GPU sorting library, which you will need to download and compile. 

What Works and What Doesn't
-------------------------------------
The makefile will compile all the code. See nVidia's nvcc guide for more compiler flags you can pass. 

Right now hamiltonian and lanczos should work to deliver the correct eigenvalues. lanczos applies reorthogonalization to prevent eigenstates from collapsing into each other. 

Right now, I am working on extending the code to work with the one dimensional transverse field Ising model. This is nearly working. 

To Do
--------------------------------------
* Add more model possibilities to hamiltonian
* Change the way lanczos looks for convergence
* Improve the diagonalization scheme in lanczos
* Add benchmarking information

2D Heisenberg Model on GPU
==========================

Author: Katharine Hyatt (kshyatt AT uwaterloo DOT ca)
Date: May 03 2012

Introduction
-------------------
This code is intended to simulate the 2D Heisenberg, 2D XY, and 1D transverse field Ising models. The 2D models are simulated on variety of square <a href="http://arxiv.org/pdf/cond-mat/0608507">Betts</a> lattices. 
It generates the sparse Hamiltonian in COO format and then applies the Lanczos method to it. 

The GPU code is based off of CPU code written by Roger Melko. 

Requirements
------------------
Since this code is CUDA based, you will need an nVidia GPU as well as the latest version of the CUDA toolkit and developer driver. The Hamiltonian-generating code also relies on Sean Baxter's <a href="http://github.com/seanbaxter/mgpu/sort">Modern GPU sorting library</a>, which you will need to download and compile. 

What Works and What Doesn't
-------------------------------------
The makefile will compile all the code. See nVidia's nvcc guide for more compiler flags you can pass. 

Right now hamiltonian and lanczos should work to deliver the correct eigenvaluesfor the Heisenberg and XY models. The verdict on the transverse field Ising model is still out. lanczos applies reorthogonalization to prevent eigenstates from collapsing into each other. 

If you want to vary the parameters, you will need to modify launcher.cu. For one parameter models (Heisenberg and XY), you only need to change J1. For the TFIM,the quantity usually called h is denoted J2 here. 

To Do
--------------------------------------
* Change the way lanczos looks for convergence
* Improve the diagonalization scheme in lanczos
* Add benchmarking information
* Improve makefile to allow release and debug compilation
* Add additional geometries to lattice

Spin Models on GPU
==========================

Author: Katharine Hyatt (kshyatt AT uwaterloo DOT ca)
Date: May 03 2012

Introduction
-------------------
This code is intended to simulate the 1D and 2D Heisenberg, 2D XY, and 1D and 2D transverse field Ising models. The 2D models are simulated on variety of square <a href="http://arxiv.org/pdf/cond-mat/0608507">Betts</a> lattices. 
It generates the sparse Hamiltonian in COO format and then applies the Lanczos method to it. 

The GPU code is based off of CPU code written by Roger Melko. 

Requirements
------------------
Since this code is CUDA based, you will need an nVidia GPU as well as the latest version of the CUDA toolkit and developer driver. The Hamiltonian-generating code also relies on Sean Baxter's <a href="http://github.com/seanbaxter/mgpu">Modern GPU library</a>, which you will need to download and compile. 


Installation
------------------------
Make sure that the CUDA toolkit and developer divers are installed following the guide <a href="http://developer.nvidia.com/nvidia-gpu-computing-documentation">here</a>.  Then:

* git clone https://github.com/kshyatt/Spin-Models-on-GPU.git
* cd Spin-Models-on-GPU
* ./setup

For later builds you can simply type "make".  If setup does not work you may have to manually download the MGPU library from the above link.

What Works and What Doesn't
-------------------------------------
The makefile will compile all the code. See nVidia's nvcc guide for more compiler flags you can pass. 

Right now hamiltonian and lanczos should work to deliver the correct eigenvalues for the Heisenberg, TF Ising, and XY models. Lanczos applies reorthogonalization to prevent eigenstates from collapsing into each other. 

To pass parameters, you will need to create data.dat. This file contains information that launcher.cu reads in to pass to functions. This way, you'll only need to compile when you make changes to the code itself. Each line in data.dat corresponds to one set of parameters, so if you're simulating five systems, data.dat should have five lines. On each line, you should put:

Number of sites | Sz sector considered (ignored for TFIM) | First coupling parameter (J1) | Second coupling parameter (J2) | Model considered | System dimension

For one parameter models (Heisenberg and XY), you only need to change J1. For the TFIM,the quantity usually called h is denoted J2 here. 

Testing
--------------------------------------
The tests folder contains code which uses Google's C++ testing framework to run some unit tests on hamiltonian and lanczos. Follow gtest's README for compilation instructions. gtest is available as a package through most Linux distros. 

To Do
--------------------------------------
* Change the way lanczos looks for convergence
* Improve the diagonalization scheme in lanczos
* Add benchmarking information
* Improve makefile to allow testing
* Add additional geometries to lattice
* Add more unit tests, especially to graphs

# Weighted Jacobi Iteration using CUDA

## Overview
This is a variant of the Jacobi Iteration in order to simulate the traversal of heat through a metal plate. The plate is represented by a 2D array of floats in which the border cells are considered to be of fixed temperature, and the interior cells have a variable temperature that is influenced by neighboring cells.

This implementation of Jacobi Iteration just visits each of the interior cells and computes a new value by computing a weighted average of the 8 surrounding cells. The process is repeated over a user specified number of passes, after which the resulting array values may be displayed as text

### Compiling and Debugging
The repo contains a Makefile that will help build the code. As it stands, you should be able to type “make” and have an executable named jacobi automatically produced for you. The make command can also take arguments. Typing “make clean” will remove all executable and object files associated with the project.
The makefile is setup to enable debugging by default, compiling with the -g and -G flags. This allows you to use the cuda debugger (called cuda-gdb), which is a modified form of gdb. Nvidia maintains documentation on this tool at their website.

## Additional Setup
After a successful build, before you run the program, you are required to set up the environment variable by using the following command in your terminal on our GPU server, or you can add the command into your .bashrc file.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib

## Running
You can run jacobi with the following arguments, where `p` determines whether results are printed, and `width` and `height` determine the size of the arrays used:
./jacobi threadsperblock passes width height [p]
If threadsperblock is zero, the serial version is run. Otherwise, this argument determines the number of threads in each block for the GPU kernel.

## Kernel0 vs Kernel1
Kernel0 works correctly but accesses global memory freely, eliminating much of the performance benefit over the serial case. 
Kernel1 reads the data values necessary for a block into shared memory from the slower global memory. Once this is done the calculations are done by reading from shared memory, and then writing results back to global memory.

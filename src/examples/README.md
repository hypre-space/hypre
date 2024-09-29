# HYPRE Examples

This directory contains example codes demonstrating the usage of HYPRE, a library for solving large, sparse linear systems of equations on massively parallel computers.

## Requirements

1. The examples are designed to be built independently of HYPRE.
2. Each example is well-documented with references to the user manual.
3. These examples mimic application codes, serving as starting templates for users.

## Building and Running

To build the examples:

```bash
cd $HYPRE_DIR/src/examples
make
```

To run an example (ex02):

```bash
mpirun -np 2 ./ex02
```

## Example Descriptions

| Example | Interface | Problem Type | Solver(s) |
|---------|-----------|--------------|-----------|
| ex01 | Struct | Basic demonstration | - |
| ex02 | Struct | 2D Poisson | SMG-PCG |
| ex03 | Struct | NxN 2D Poisson | SMG, SMG-PCG |
| ex04 | Struct | NxN 2D Convection-Reaction-Diffusion | SMG/PFMG - CG/GMRES |
| ex05 | IJ | NxN 2D Poisson | AMG-PCG |
| ex06 | SStruct | 2D Poisson | SMG-PCG |
| ex07 | SStruct | NxN 2D Convection-Reaction-Diffusion | SMG/PFMG - CG/GMRES |
| ex08 | SStruct | 3-parts 2D Poisson | Split-PCG |
| ex09 | SStruct | 2D biharmonic | sysPFMG |
| ex11 | IJ | NxN 2D Poisson eigenvalue problem | AMG-LOBPCG |
| ex12 | SStruct | Nodal 2D Poisson | PFMG, AMG |
| ex13 | SStruct | 6-parts 2D Poisson | AMG |
| ex14 | SStruct | 6-parts 2D Poisson SStruct FEM | AMG |
| ex15 | SStruct | 3D electromagnetic diffusion | AMS |
| ex16 | SStruct | 2D high order Q3 FEM Poisson | AMG |
| ex17 | Struct | 4D Poisson | CG |
| ex18 | SStruct | 4D Poisson | CG |

## GPU Support

GPU support is available for CUDA, HIP, and SYCL. To use GPU acceleration:

1. Ensure HYPRE is built with the appropriate GPU backend
2. Set the corresponding environment variables (e.g., CUDA_HOME for CUDA)
3. Use the `gpu` target when building: `make gpu`

Note:
1. Running examples on GPUs requires that hypre is built with Unified Memory.
2. Current GPU usage in these examples is suboptimal due to arrays being allocated in small sizes for setting up matrix coefficients per row. For better GPU utilization, matrices and vectors should be filled in larger blocks of data.

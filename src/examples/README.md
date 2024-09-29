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

1. **ex01**: Basic example demonstrating the use of HYPRE's struct interface.
2. **ex02**: Struct interface - 2D Poisson - SMG-PCG solver.
3. **ex03**: Struct interface - NxN 2D Poisson - SMG and SMG-PCG solvers.
4. **ex04**: Struct interface - NxN 2D Convection-Reaction-Diffusion.
5. **ex05**: IJ interface - NxN 2D Poisson - AMG-PCG solver.
6. **ex06**: SStruct interface - 2D Poisson - SMG-PCG solver.
7. **ex07**: SStruct interface - NxN 2D Convection-Reaction-Diffusion.
8. **ex08**: SStruct interface - 3-parts 2D Poisson - Split-PCG solver.
9. **ex09**: SStruct interface - 2D biharmonic - sysPFMG solver.
10. **ex11**: IJ interface - NxN 2D Poisson eigenvalue problem - AMG-LOBPCG solver.
11. **ex12**: SStruct interface - Nodal 2D Poisson - PFMG and AMG solvers.
12. **ex13**: SStruct interface - 6-parts 2D Poisson - AMG solver.
13. **ex14**: SStruct interface - 6-parts 2D Poisson SStruct FEM - AMG solver.
14. **ex15**: SStruct interface - 3D electromagnetic diffusion - AMS solver.
15. **ex16**: SStruct interface - 2D high order Q3 FEM Poisson - AMG solver.
15. **ex17**: Struct interface - 4D Poisson - CG solver.
15. **ex18**: SStruct interface - 4D Poisson - CG solver.
15. **ex18comp**: SStruct interface - Complex 4D Poisson - CG solver.

## GPU Support

GPU support is available for CUDA, HIP, and SYCL. To use GPU acceleration:

1. Ensure HYPRE is built with the appropriate GPU backend
2. Set the corresponding environment variables (e.g., CUDA_HOME for CUDA)
3. Use the `gpu` target when building: `make gpu`

Note:
1. Running examples on GPUs requires that hypre is built with Unified Memory.
2. Current GPU usage in these examples is suboptimal due to arrays being allocated in small sizes for setting up matrix coefficients per row. For better GPU utilization, matrices and vectors should be filled in larger blocks of data.

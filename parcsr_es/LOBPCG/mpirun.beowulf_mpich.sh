#!/bin/sh
# This script compares PCG with LOBPCG and runs different LOBPCG tests
# 
#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ./hypre_arch.sh

MPIRUN="/opt/mpich/bin/mpirun.$HYPRE_ARCH"
#MPIRUN="./mpirun.$HYPRE_ARCH"
DRIVER="./IJ_eigen_solver"

NUMBER_OF_CPU=2

#=============================================================================
# IJ_linear_and_eigenvaluesolvers: (-nolobpcg option runs PCG)
# Run default case with different preconditioners (solvers) for PCG and LOBPCG:  
#    1:  BoomerAMG_PCG
#    2:  DS_PCG
#    8:  ParaSails_PCG
#    12: Schwarz-PCG  
#    43: Euclid-PCG
#=============================================================================
#
# PCG run:
$MPIRUN -np $NUMBER_OF_CPU $DRIVER  -nolobpcg -solver 1 
#LOBPCG run for one eigenpair:
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 1 -xrand 1 
#LOBPCG run for several eigenpairs:
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 1 -xrand 5 

$MPIRUN -np $NUMBER_OF_CPU $DRIVER  -nolobpcg -solver 2 
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 2 -xrand 1
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 2 -xrand 5

$MPIRUN -np $NUMBER_OF_CPU $DRIVER  -nolobpcg -solver 8 
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 8 -xrand 1
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 8 -xrand 5

$MPIRUN -np $NUMBER_OF_CPU $DRIVER  -nolobpcg -solver 12 
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 12 -xrand 1
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 12 -xrand 5

#$MPIRUN -np 2 $DRIVER  -nolobpcg -solver 43 -rhsrand
$MPIRUN -np $NUMBER_OF_CPU $DRIVER  -nolobpcg -solver 43
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 43 -xrand 1
$MPIRUN -np $NUMBER_OF_CPU $DRIVER            -solver 43 -xrand 5

#more tests for LOBPCG only 
#
#same problem and solver with different number of eigenvectors computed
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -9pt -solver 12  -pcgitr 2 -xrand 1
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -9pt -solver 12  -pcgitr 2 -xrand 2
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -9pt -solver 12  -pcgitr 2 -xrand 4
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -9pt -solver 12  -pcgitr 2 -xrand 8

#same problem and solver with different number of inner iterations 
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -27pt -solver 1 -xrand 5 #-pcgitr 1 #this is the default
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -27pt -solver 1 -xrand 5 -pcgitr 2
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -27pt -solver 1 -xrand 5 -pcgitr 3

#the next 3 runs must produce identical results
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -laplacian -n 10 10 10 -solver 43 -xin Xin.mtx -pcgitr 2
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -ain laplacian_10_10_10.mtx -solver 43  -xin Xin.mtx -pcgitr 2
$MPIRUN -np $NUMBER_OF_CPU $DRIVER -ain laplacian_10_10_10.mtx -tin laplacian_10_10_10.mtx -solver 43 -xin Xin.mtx -pcgitr 2

# some really big ones:
#mpirun -np $NUMBER_OF_CPU $DRIVER -27pt -n  50  50  50 -solver 1 -xrand 40 -pcgitr 2
#mpirun -np $NUMBER_OF_CPU $DRIVER -27pt -n 100 100 100 -solver 2 -xrand  4 -pcgitr 2

#!/bin/sh
# This script compares PCG with LOBPCG and runs different LOBPCG tests
#===========================================================================

MPIRUN="poe"
RUNPARM="-nodes 1 -procs 4 -rmpool 0"
#See http://www.llnl.gov/asci/platforms/bluepac/cpu_use.html

DRIVER="./ij_es"

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
$MPIRUN $DRIVER  -solver 1 $RUNPARM 
#LOBPCG run for one eigenpair:
$MPIRUN $DRIVER  -lobpcg  -solver 1 -vrand 1 $RUNPARM 
#LOBPCG run for several eigenpairs:
$MPIRUN $DRIVER  -lobpcg -solver 1 -vrand 5 $RUNPARM 

$MPIRUN $DRIVER  -solver 2 
$MPIRUN $DRIVER  -lobpcg -solver 2 -vrand 1 $RUNPARM
$MPIRUN $DRIVER  -lobpcg -solver 2 -vrand 5 $RUNPARM

$MPIRUN $DRIVER  -solver 8 $RUNPARM 
$MPIRUN $DRIVER  -lobpcg -solver 8 -vrand 1 $RUNPARM
$MPIRUN $DRIVER  -lobpcg -solver 8 -vrand 5 $RUNPARM

$MPIRUN $DRIVER  -solver 12 $RUNPARM 
$MPIRUN $DRIVER  -lobpcg -solver 12 -vrand 1 $RUNPARM
$MPIRUN $DRIVER  -lobpcg -solver 12 -vrand 5 $RUNPARM

$MPIRUN $DRIVER  -solver 43 $RUNPARM
$MPIRUN $DRIVER  -lobpcg -solver 43 -vrand 1 $RUNPARM
$MPIRUN $DRIVER  -lobpcg -solver 43 -vrand 5 $RUNPARM

#more tests for LOBPCG only 
#
#same problem and solver with different number of eigenvectors computed
$MPIRUN $DRIVER -9pt -lobpcg -solver 12  -pcgitr 2 -vrand 1 $RUNPARM
$MPIRUN $DRIVER -9pt -lobpcg -solver 12  -pcgitr 2 -vrand 2 $RUNPARM
$MPIRUN $DRIVER -9pt -lobpcg -solver 12  -pcgitr 2 -vrand 4 $RUNPARM
$MPIRUN $DRIVER -9pt -lobpcg -solver 12  -pcgitr 2 -vrand 8 $RUNPARM

#same problem and solver with different number of inner iterations 
$MPIRUN $DRIVER -27pt -lobpcg -solver 1 -vrand 5 $RUNPARM #-pcgitr 1 #this is the default
$MPIRUN $DRIVER -27pt -lobpcg -solver 1 -vrand 5 -pcgitr 2 $RUNPARM
$MPIRUN $DRIVER -27pt -lobpcg -solver 1 -vrand 5 -pcgitr 3 $RUNPARM

#the next 3 runs must produce identical results
$MPIRUN $DRIVER -laplacian -n 10 10 10 -lobpcg -solver 43 -vin Xin.mtx -pcgitr 2 $RUNPARM
$MPIRUN $DRIVER -lobpcg -ain laplacian_10_10_10.mtx -solver 43  -vin Xin.mtx -pcgitr 2 $RUNPARM
$MPIRUN $DRIVER -lobpcg -ain laplacian_10_10_10.mtx -tin laplacian_10_10_10.mtx -solver 43 -vin Xin.mtx -pcgitr 2 $RUNPARM

# some really big ones:
#mpirun $DRIVER -lobpcg -27pt -n  50  50  50 -solver 1 -vrand 40 -pcgitr 2 $RUNPARM
#mpirun $DRIVER -lobpcg -27pt -n 100 100 100 -solver 2 -vrand  4 -pcgitr 2 $RUNPARM 

#!/bin/sh

#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ../../../test/hypre_arch.sh
MPIRUN="../../../test/mpirun.$HYPRE_ARCH"
DRIVER="../../../test/IJ_linear_solvers"

#=================================================================
# single cpu tests
#=================================================================

rm -rf *.out* *temp *database

#
#read options from "database" (the default configuration filename)
#
cp input/level.3 ./database
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData eu.temp
diff eu.temp  output/test3.out >&2
rm -f database eu.temp

#
#specify a different configuration filename, and read options therefrom
#
cp input/level.3 .
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -db_filename level.3 -printTestData eu.temp
diff eu.temp output/test3.out >&2
rm -f level.3 eu.temp

#
#ensure command line options override options in the configuration file
#
cp input/level.3  ./database
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData eu.temp
diff eu.temp output/test2.out >&2
rm -f database eu.temp


#=================================================================
# mulitple cpu tests
#=================================================================


rm -rf *.out* *temp *database

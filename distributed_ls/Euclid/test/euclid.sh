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

#cp input/test3.options ./database
#$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData test3.temp
#diff test3.temp  output/test3.temp >&2


cp input/test3.options ./test3.database
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -db_filename test3.database -printTestData test3.temp
diff test3.temp  output/test3.temp >&2

exit

$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData test1.temp
diff test1.temp  output/test1.temp >&2

$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData -level 3 test2.temp
diff test2.temp  output/test2.temp >&2




#=================================================================
# mulitple cpu tests
#=================================================================

#! /usr/local/bin/tcsh -f

#===========================================================================
#   runs test drivers and logs results to file
#===========================================================================

HYPRE_AUTOTEST_LOG_FILE="autotest.log"

rm -f $HYPRE_AUTOTEST_LOG.FILE
./struct_linear_solvers >> $HYPRE_AUTOTEST_LOG_FILE


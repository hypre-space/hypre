/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_LSI_SSOR interface
 *
 *****************************************************************************/

#ifndef __HYPRE_SSOR__
#define __HYPRE_SSOR__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/utilities.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

extern "C"
{
   int HYPRE_LSI_SSORCreate( MPI_Comm comm, HYPRE_Solver *solver );
   int HYPRE_LSI_SSORDestroy( HYPRE_Solver solver );
   int HYPRE_LSI_SSORSetNumSweeps( HYPRE_Solver solver, int num);
   int HYPRE_LSI_SSORSetOutputLevel( HYPRE_Solver solver, int level);
   int HYPRE_LSI_SSORSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,   HYPRE_ParVector x );
}

#endif


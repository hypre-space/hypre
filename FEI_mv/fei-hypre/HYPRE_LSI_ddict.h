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
 * HYPRE_DDICT interface
 *
 *****************************************************************************/

#ifndef __HYPRE_DDICT__
#define __HYPRE_DDICT__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/utilities.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

extern "C"
{
   int HYPRE_LSI_DDICTCreate( MPI_Comm comm, HYPRE_Solver *solver );
   int HYPRE_LSI_DDICTDestroy( HYPRE_Solver solver );
   int HYPRE_LSI_DDICTSetFillin( HYPRE_Solver solver, double fillin);
   int HYPRE_LSI_DDICTSetOutputLevel( HYPRE_Solver solver, int level);
   int HYPRE_LSI_DDICTSetDropTolerance( HYPRE_Solver solver, double thresh);
   int HYPRE_LSI_DDICTSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b,   HYPRE_ParVector x );
   int HYPRE_LSI_DDICTSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b,   HYPRE_ParVector x );
}

#endif


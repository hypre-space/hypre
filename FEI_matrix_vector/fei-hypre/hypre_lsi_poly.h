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
 * HYPRE_POLY interface
 *
 *****************************************************************************/

#ifndef __HYPRE_POLY__
#define __HYPRE_POLY__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/utilities.h"
#include "parcsr_linear_solvers/HYPRE_parcsr_ls.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

extern "C"
{
   int HYPRE_LSI_PolyCreate( MPI_Comm comm, HYPRE_Solver *solver );
   int HYPRE_LSI_PolyDestroy( HYPRE_Solver solver );
   int HYPRE_LSI_PolySetOrder( HYPRE_Solver solver, int order);
   int HYPRE_LSI_PolySetOutputLevel( HYPRE_Solver solver, int level);
   int HYPRE_LSI_PolySolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,   HYPRE_ParVector x );
   int HYPRE_LSI_PolySetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,   HYPRE_ParVector x );
}

#endif


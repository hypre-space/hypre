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
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif
extern int HYPRE_LSI_DDICTCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_DDICTDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_DDICTSetFillin( HYPRE_Solver solver, double fillin);
extern int HYPRE_LSI_DDICTSetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_DDICTSetDropTolerance( HYPRE_Solver solver, double thresh);
extern int HYPRE_LSI_DDICTSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                 HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_DDICTSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                 HYPRE_ParVector b,   HYPRE_ParVector x );
#ifdef __cplusplus
}
#endif

#endif


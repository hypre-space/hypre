/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_Schwarz interface
 *
 *****************************************************************************/

#ifndef __HYPRE_SCHWARZ__
#define __HYPRE_SCHWARZ__

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

extern int HYPRE_LSI_SchwarzCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_SchwarzDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_SchwarzSetBlockSize( HYPRE_Solver solver, int blksize);
extern int HYPRE_LSI_SchwarzSetNBlocks( HYPRE_Solver solver, int nblks);
extern int HYPRE_LSI_SchwarzSetILUTFillin( HYPRE_Solver solver, double fillin);
extern int HYPRE_LSI_SchwarzSetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_SchwarzSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_SchwarzSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,   HYPRE_ParVector x );

#ifdef __cplusplus
}
#endif

#endif


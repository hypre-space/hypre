/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_ls library
 *
 *****************************************************************************/

#ifndef HYPRE_LS_HEADER
#define HYPRE_LS_HEADER

#include "HYPRE_struct_pcg.h"
#include "HYPRE_pcg.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *HYPRE_StructSolver;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_smg.c */
HYPRE_StructSolver HYPRE_StructSMGInitialize P((MPI_Comm comm ));
int HYPRE_StructSMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_SMGSetMemoryUse P((HYPRE_StructSolver solver , int memory_use ));
int HYPRE_SMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_SMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_SMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_SMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_SMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_SMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_SMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

#undef P

#endif

/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
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

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *HYPRE_StructSolver;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize P((MPI_Comm comm ));
int HYPRE_AMGFinalize P((HYPRE_Solver solver ));
int HYPRE_AMGSetup P((HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSolve P((HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_AMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_AMGSetZeroGuess P((HYPRE_Solver solver ));
int HYPRE_AMGGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_AMGGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *relative_residual_norm ));
 
#undef P

#endif

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
 * Header file for ZZZ_ls library
 *
 *****************************************************************************/

#ifndef ZZZ_LS_HEADER
#define ZZZ_LS_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *ZZZ_StructSolver;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* ZZZ_struct_smg.c */
ZZZ_StructSolver ZZZ_StructSMGInitialize P((MPI_Comm *comm ));
int ZZZ_StructSMGFinalize P((ZZZ_StructSolver solver ));
int ZZZ_StructSMGSetup P((ZZZ_StructSolver solver , ZZZ_StructMatrix A , ZZZ_StructVector b , ZZZ_StructVector x ));
int ZZZ_StructSMGSolve P((ZZZ_StructSolver solver , ZZZ_StructMatrix A , ZZZ_StructVector b , ZZZ_StructVector x ));
int ZZZ_SMGSetMemoryUse P((ZZZ_StructSolver solver , int memory_use ));
int ZZZ_SMGSetTol P((ZZZ_StructSolver solver , double tol ));
int ZZZ_SMGSetMaxIter P((ZZZ_StructSolver solver , int max_iter ));
int ZZZ_SMGSetZeroGuess P((ZZZ_StructSolver solver ));
int ZZZ_SMGGetNumIterations P((ZZZ_StructSolver solver , int *num_iterations ));
int ZZZ_SMGGetFinalRelativeResidualNorm P((ZZZ_StructSolver solver , double *relative_residual_norm ));
 
#undef P

#endif

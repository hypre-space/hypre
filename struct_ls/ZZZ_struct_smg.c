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
 * ZZZ_StructSMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_StructSMGInitialize
 *--------------------------------------------------------------------------*/

ZZZ_StructSolver
ZZZ_StructSMGInitialize( MPI_Comm *comm )
{
   return ( (ZZZ_StructSolver) zzz_SMGInitialize( comm ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSMGFinalize
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSMGFinalize( ZZZ_StructSolver solver )
{
   return( zzz_SMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSMGSetup
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSMGSetup( ZZZ_StructSolver solver,
                    ZZZ_StructMatrix A,
                    ZZZ_StructVector b,
                    ZZZ_StructVector x      )
{
   return( zzz_SMGSetup( (void *) solver,
                         (zzz_StructMatrix *) A,
                         (zzz_StructVector *) b,
                         (zzz_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSMGSolve
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSMGSolve( ZZZ_StructSolver solver,
                    ZZZ_StructMatrix A,
                    ZZZ_StructVector b,
                    ZZZ_StructVector x      )
{
   return( zzz_SMGSolve( (void *) solver,
                         (zzz_StructMatrix *) A,
                         (zzz_StructVector *) b,
                         (zzz_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SMGSetMemoryUse
 *--------------------------------------------------------------------------*/

int
ZZZ_SMGSetMemoryUse( ZZZ_StructSolver solver,
                     int              memory_use )
{
   return( zzz_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SMGSetTol
 *--------------------------------------------------------------------------*/

int
ZZZ_SMGSetTol( ZZZ_StructSolver solver,
               double           tol    )
{
   return( zzz_SMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
ZZZ_SMGSetMaxIter( ZZZ_StructSolver solver,
                   int              max_iter  )
{
   return( zzz_SMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
ZZZ_SMGSetZeroGuess( ZZZ_StructSolver solver )
{
   return( zzz_SMGSetZeroGuess( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
ZZZ_SMGGetNumIterations( ZZZ_StructSolver  solver,
                         int              *num_iterations )
{
   return( zzz_SMGGetNumIterations( (void *) solver, num_iterations ) );
}


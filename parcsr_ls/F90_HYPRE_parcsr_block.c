/*BHEADER**********************************************************************
 * (c) 2005   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_BlockTridiag Fortran interface
 *
 *****************************************************************************/

#include "block_tridiag.h"
#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagcreate, HYPRE_BLOCKTRIDIAGCREATE)
               (long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagCreate( (HYPRE_Solver *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, HYPRE_BLOCKTRIDIAGDESTROY)
               (long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagDestroy( (HYPRE_Solver) *solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, HYPRE_BLOCKTRIDIAGSETUP)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetup( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, HYPRE_BLOCKTRIDIAGSOLVE)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSolve( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, HYPRE_BLOCKTRIDIAGSETINDEXSET)
               (long int *solver, int *n, int *inds, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetIndexSet( (HYPRE_Solver) *solver,
                                                (int)          *n, 
                                                (int *)         inds);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
               (long int *solver, double *thresh, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGStrengthThreshold( (HYPRE_Solver) *solver,
                                                            (double)       *thresh);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
               (long int *solver, int *num_sweeps, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGNumSweeps( (HYPRE_Solver) *solver,
                                                    (int)          *num_sweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
               (long int *solver, int *relax_type, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGRelaxType( (HYPRE_Solver) *solver,
                                                    (int)          *relax_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
               (long int *solver, int *print_level, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetPrintLevel( (HYPRE_Solver) *solver,
                                                  (int)          *print_level);
}

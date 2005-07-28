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
 * HYPRE_Schwarz Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzcreate, HYPRE_SCHWARZCREATE)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzCreate( (HYPRE_Solver *) solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzdestroy, HYPRE_SCHWARZDESTROY)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzDestroy( (HYPRE_Solver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetup, HYPRE_SCHWARZSETUP)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSetup( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsolve, HYPRE_SCHWARZSOLVE)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSolve( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_schwarzsetvariant, HYPRE_SCHWARZSETVARIANT)
               (long int *solver, int *variant, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSetVariant( (HYPRE_Solver) *solver,
                                            (int)          *variant ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetoverlap, HYPRE_SCHWARZSETOVERLAP)
               (long int *solver, int *overlap, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSetOverlap( (HYPRE_Solver) *solver, 
                                            (int)          *overlap));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomaintype, HYPRE_SCHWARZSETDOMAINTYPE)
               (long int *solver, int *domain_type, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSetDomainType( (HYPRE_Solver) *solver,
                                               (int)          *domain_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomainstructure, HYPRE_SCHWARZSETDOMAINSTRUCTURE)
               (long int *solver, long int *domain_structure, int *ierr)
{
   *ierr = (int) ( HYPRE_SchwarzSetDomainStructure( (HYPRE_Solver)    *solver,
                                                    (HYPRE_CSRMatrix) *domain_structure));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetnumfunctions, HYPRE_SCHWARZSETNUMFUNCTIONS)
               (long int *solver, int *num_functions, int *ierr)
{
   *ierr = (int) (HYPRE_SchwarzSetNumFunctions( (HYPRE_Solver) *solver,
                                                (int)          *num_functions ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetrelaxweight, HYPRE_SCHWARZSETRELAXWEIGHT)
               (long int *solver, double *relax_weight, int *ierr)
{
   *ierr = (int) (HYPRE_SchwarzSetRelaxWeight( (HYPRE_Solver) *solver,
                                               (double)       *relax_weight));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdoffunc, HYPRE_SCHWARZSETDOFFUNC)
               (long int *solver, int *dof_func, int *ierr)
{
   *ierr = (int) (HYPRE_SchwarzSetDofFunc( (HYPRE_Solver) *solver,
                                           (int *)         dof_func  ));
}

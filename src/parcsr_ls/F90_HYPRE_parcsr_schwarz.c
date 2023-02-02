/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Schwarz Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzcreate, HYPRE_SCHWARZCREATE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzdestroy, HYPRE_SCHWARZDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetup, HYPRE_SCHWARZSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsolve, HYPRE_SCHWARZSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_schwarzsetvariant, HYPRE_SCHWARZSETVARIANT)
(hypre_F90_Obj *solver,
 hypre_F90_Int *variant,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSetVariant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (variant) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetoverlap, HYPRE_SCHWARZSETOVERLAP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *overlap,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSetOverlap(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (overlap)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomaintype, HYPRE_SCHWARZSETDOMAINTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *domain_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSetDomainType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (domain_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomainstructure, HYPRE_SCHWARZSETDOMAINSTRUCTURE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *domain_structure,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SchwarzSetDomainStructure(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_CSRMatrix, domain_structure)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetnumfunctions, HYPRE_SCHWARZSETNUMFUNCTIONS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_functions,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetNumFunctions(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (num_functions) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetrelaxweight, HYPRE_SCHWARZSETRELAXWEIGHT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *relax_weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetRelaxWeight(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassReal (relax_weight)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdoffunc, HYPRE_SCHWARZSETDOFFUNC)
(hypre_F90_Obj *solver,
 hypre_F90_IntArray *dof_func,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetDofFunc(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassIntArray (dof_func)  ));
}
#ifdef __cplusplus
}
#endif

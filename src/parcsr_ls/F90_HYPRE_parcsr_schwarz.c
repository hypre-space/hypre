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
#include "_hypre_fortran.h"

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

void
hypre_F90_IFACE(hypre_schwarzsetlocalsolvertype, HYPRE_SCHWARZSETLOCALSOLVERTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *local_solver_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetLocalSolverType(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (local_solver_type)));
}

void
hypre_F90_IFACE(hypre_schwarzsetilukleveloffill, HYPRE_SCHWARZSETILUKLEVELOFFILL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *level_of_fill,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetILUKLevelOfFill(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (level_of_fill)));
}

void
hypre_F90_IFACE(hypre_schwarzsetilutmaxnnzperrow, HYPRE_SCHWARZSETILUTMAXNNZPERROW)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_nnz_row,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetILUTMaxNnzPerRow(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (max_nnz_row)));
}

void
hypre_F90_IFACE(hypre_schwarzsetilutdroptol, HYPRE_SCHWARZSETILUTDROPTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *droptol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetILUTDroptol(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassReal (droptol)));
}

void
hypre_F90_IFACE(hypre_schwarzsetmaxiter, HYPRE_SCHWARZSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetMaxIter(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (max_iter)));
}

void
hypre_F90_IFACE(hypre_schwarzsettol, HYPRE_SCHWARZSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetTol(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassReal (tol)));
}

void
hypre_F90_IFACE(hypre_schwarzsetprintlevel, HYPRE_SCHWARZSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetPrintLevel(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (print_level)));
}

void
hypre_F90_IFACE(hypre_schwarzsetlogging, HYPRE_SCHWARZSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzSetLogging(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassInt (logging)));
}

void
hypre_F90_IFACE(hypre_schwarzgetnumiterations, HYPRE_SCHWARZGETNUMITERATIONS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzGetNumIterations(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassIntRef (num_iterations)));
}

void
hypre_F90_IFACE(hypre_schwarzgetfinalresidualnorm, HYPRE_SCHWARZGETFINALRESIDUALNORM)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SchwarzGetFinalResidualNorm(
               hypre_F90_PassObj (HYPRE_Solver, solver),
               hypre_F90_PassRealRef (norm)));
}
#ifdef __cplusplus
}
#endif

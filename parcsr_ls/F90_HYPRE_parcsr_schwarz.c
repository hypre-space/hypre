/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzCreate( (HYPRE_Solver *) solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzdestroy, HYPRE_SCHWARZDESTROY)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzDestroy( (HYPRE_Solver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetup, HYPRE_SCHWARZSETUP)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSetup( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsolve, HYPRE_SCHWARZSOLVE)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSolve( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_schwarzsetvariant, HYPRE_SCHWARZSETVARIANT)
               (hypre_F90_Obj *solver, HYPRE_Int *variant, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSetVariant( (HYPRE_Solver) *solver,
                                            (HYPRE_Int)          *variant ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetoverlap, HYPRE_SCHWARZSETOVERLAP)
               (hypre_F90_Obj *solver, HYPRE_Int *overlap, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSetOverlap( (HYPRE_Solver) *solver, 
                                            (HYPRE_Int)          *overlap));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomaintype, HYPRE_SCHWARZSETDOMAINTYPE)
               (hypre_F90_Obj *solver, HYPRE_Int *domain_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSetDomainType( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *domain_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomainstructure, HYPRE_SCHWARZSETDOMAINSTRUCTURE)
               (hypre_F90_Obj *solver, hypre_F90_Obj *domain_structure, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SchwarzSetDomainStructure( (HYPRE_Solver)    *solver,
                                                    (HYPRE_CSRMatrix) *domain_structure));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetnumfunctions, HYPRE_SCHWARZSETNUMFUNCTIONS)
               (hypre_F90_Obj *solver, HYPRE_Int *num_functions, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SchwarzSetNumFunctions( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *num_functions ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetrelaxweight, HYPRE_SCHWARZSETRELAXWEIGHT)
               (hypre_F90_Obj *solver, double *relax_weight, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SchwarzSetRelaxWeight( (HYPRE_Solver) *solver,
                                               (double)       *relax_weight));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdoffunc, HYPRE_SCHWARZSETDOFFUNC)
               (hypre_F90_Obj *solver, HYPRE_Int *dof_func, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SchwarzSetDofFunc( (HYPRE_Solver) *solver,
                                           (HYPRE_Int *)         dof_func  ));
}

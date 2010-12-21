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
 * HYPRE_ParCSRPilut Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutcreate, HYPRE_PARCSRPILUTCREATE)( hypre_F90_Comm *comm,
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutCreate( (MPI_Comm)       *comm,
                                                (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutdestroy, HYPRE_PARCSRPILUTDESTROY)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetup, HYPRE_PARCSRPILUTSETUP)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutSetup( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsolve, HYPRE_PARCSRPILUTSOLVE)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutSolve( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetmaxiter, HYPRE_PARCSRPILUTSETMAXITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *max_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutSetMaxIter( (HYPRE_Solver) *solver, 
                                                (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropToleran
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetdroptoleran, HYPRE_PARCSRPILUTSETDROPTOLERAN)( hypre_F90_Obj *solver,
                                                  double   *tol,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutSetDropTolerance( (HYPRE_Solver) *solver, 
                                                      (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFacRowSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetfacrowsize, HYPRE_PARCSRPILUTSETFACROWSIZE)( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *size,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPilutSetFactorRowSize( (HYPRE_Solver) *solver,
                                                      (HYPRE_Int)          *size    ) );
}


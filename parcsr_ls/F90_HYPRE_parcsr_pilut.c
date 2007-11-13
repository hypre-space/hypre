/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
hypre_F90_IFACE(hypre_parcsrpilutcreate, HYPRE_PARCSRPILUTCREATE)( int      *comm,
                                              long int *solver,
                                              int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPilutCreate( (MPI_Comm)       *comm,
                                                (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutdestroy, HYPRE_PARCSRPILUTDESTROY)( long int *solver,
                                            int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPilutDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetup, HYPRE_PARCSRPILUTSETUP)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetup( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsolve, HYPRE_PARCSRPILUTSOLVE)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSolve( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetmaxiter, HYPRE_PARCSRPILUTSETMAXITER)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetMaxIter( (HYPRE_Solver) *solver, 
                                                (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropToleran
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetdroptoleran, HYPRE_PARCSRPILUTSETDROPTOLERAN)( long int *solver,
                                                  double   *tol,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetDropTolerance( (HYPRE_Solver) *solver, 
                                                      (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFacRowSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetfacrowsize, HYPRE_PARCSRPILUTSETFACROWSIZE)( long int *solver,
                                                 int      *size,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetFactorRowSize( (HYPRE_Solver) *solver,
                                                      (int)          *size    ) );
}


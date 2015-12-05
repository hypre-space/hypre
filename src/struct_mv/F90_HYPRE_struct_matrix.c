/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixcreate, HYPRE_STRUCTMATRIXCREATE)( int      *comm,
                                           long int *grid,
                                           long int *stencil,
                                           long int *matrix,
                                           int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructMatrixCreate( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructStencil)  *stencil,
                                  (HYPRE_StructMatrix *) matrix   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixdestroy, HYPRE_STRUCTMATRIXDESTROY)( long int *matrix,
                                            int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixDestroy( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixinitialize, HYPRE_STRUCTMATRIXINITIALIZE)( long int *matrix,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixInitialize( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetvalues, HYPRE_STRUCTMATRIXSETVALUES)( long int *matrix,
                                              int      *grid_index,
                                              int      *num_stencil_indices,
                                              int      *stencil_indices,
                                              double   *values,
                                              int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetValues( (HYPRE_StructMatrix) *matrix,
                                     (int *)              grid_index,
                                     (int)                *num_stencil_indices,
                                     (int *)              stencil_indices,
                                     (double *)           values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetboxvalues, HYPRE_STRUCTMATRIXSETBOXVALUES)( long int *matrix,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 int      *num_stencil_indices,
                                                 int      *stencil_indices,
                                                 double   *values,
                                                 int      *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (int)                *num_stencil_indices,
                                        (int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixgetboxvalues, HYPRE_STRUCTMATRIXGETBOXVALUES)( long int *matrix,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 int      *num_stencil_indices,
                                                 int      *stencil_indices,
                                                 double   *values,
                                                 int      *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixGetBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (int)                *num_stencil_indices,
                                        (int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetconstantva, HYPRE_STRUCTMATRIXSETCONSTANTVA)( long int *matrix,
                                                int      *num_stencil_indices,
                                                int      *stencil_indices,
                                                double   *values,
                                                int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetConstantValues( (HYPRE_StructMatrix) *matrix,
                                             (int)       *num_stencil_indices,
                                             (int *)     stencil_indices,
                                             (double *)  values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtovalues, HYPRE_STRUCTMATRIXADDTOVALUES)( long int *matrix,
                                                int      *grid_index,
                                                int      *num_stencil_indices,
                                                int      *stencil_indices,
                                                double   *values,
                                                int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAddToValues( (HYPRE_StructMatrix) *matrix,
                                       (int *)     grid_index,
                                       (int)       *num_stencil_indices,
                                       (int *)     stencil_indices,
                                       (double *)  values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtoboxvalues, HYPRE_STRUCTMATRIXADDTOBOXVALUES)( long int *matrix,
                                                   int    *ilower,
                                                   int    *iupper,
                                                   int    *num_stencil_indices,
                                                   int    *stencil_indices,
                                                   double *values,
                                                   int    *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAddToBoxValues( (HYPRE_StructMatrix) *matrix,
                                          (int *)     ilower,
                                          (int *)     iupper,
                                          (int)       *num_stencil_indices,
                                          (int *)     stencil_indices,
                                          (double *)  values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtoconstant, HYPRE_STRUCTMATRIXADDTOCONSTANT)( long int *matrix,
                                                   int    *num_stencil_indices,
                                                   int    *stencil_indices,
                                                   double *values,
                                                   int    *ierr              )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetConstantValues( (HYPRE_StructMatrix) *matrix,
                                             (int)       *num_stencil_indices,
                                             (int *)     stencil_indices,
                                             (double *)  values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixassemble, HYPRE_STRUCTMATRIXASSEMBLE)( long int *matrix,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixAssemble( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetnumghost, HYPRE_STRUCTMATRIXSETNUMGHOST)( long int *matrix,
                                                int      *num_ghost,
                                                int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetNumGhost( (HYPRE_StructMatrix) *matrix,
                                       (int *)               num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetgrid, HYPRE_STRUCTMATRIXGETGRID)( long int *matrix,
                                            long int *grid,
                                            int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructMatrixGetGrid( (HYPRE_StructMatrix) *matrix,
                                (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetsymmetric, HYPRE_STRUCTMATRIXSETSYMMETRIC)( long int *matrix,
                                                 int      *symmetric,
                                                 int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetSymmetric( (HYPRE_StructMatrix) *matrix,
                                        (int)                *symmetric ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantEntries
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetconstanten, HYPRE_STRUCTMATRIXSETCONSTANTEN)( long int *matrix,
                                                int      *nentries,
                                                int      *entries,
                                                int      *ierr                )
{
   *ierr = (int)
      ( HYPRE_StructMatrixSetConstantEntries( (HYPRE_StructMatrix) *matrix,
                                              (int)       *nentries,
                                              (int *)     entries           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixprint, HYPRE_STRUCTMATRIXPRINT)(
   long int  *matrix,
   int       *all,
   int       *ierr )
{
   *ierr = (int)
      ( HYPRE_StructMatrixPrint("HYPRE_StructMatrix.out",
                                (HYPRE_StructMatrix) *matrix,
                                (int)                *all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixmatvec, HYPRE_STRUCTMATRIXMATVEC)
                                              ( double   *alpha,
                                                long int *A,
                                                long int *x,
                                                double   *beta,
                                                long int *y,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_StructMatrixMatvec( (double)             *alpha,
                                             (HYPRE_StructMatrix) *A,
                                             (HYPRE_StructVector) *x,
                                             (double)             *beta,
                                             (HYPRE_StructVector) *y  ) );
}

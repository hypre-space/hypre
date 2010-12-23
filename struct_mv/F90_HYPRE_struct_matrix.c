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
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixcreate, HYPRE_STRUCTMATRIXCREATE)( hypre_F90_Comm *comm,
                                           hypre_F90_Obj *grid,
                                           hypre_F90_Obj *stencil,
                                           hypre_F90_Obj *matrix,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixCreate( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructStencil)  *stencil,
                                  (HYPRE_StructMatrix *) matrix   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixdestroy, HYPRE_STRUCTMATRIXDESTROY)( hypre_F90_Obj *matrix,
                                            HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixDestroy( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixinitialize, HYPRE_STRUCTMATRIXINITIALIZE)( hypre_F90_Obj *matrix,
                                               HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixInitialize( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetvalues, HYPRE_STRUCTMATRIXSETVALUES)( hypre_F90_Obj *matrix,
                                              HYPRE_Int      *grid_index,
                                              HYPRE_Int      *num_stencil_indices,
                                              HYPRE_Int      *stencil_indices,
                                              double   *values,
                                              HYPRE_Int      *ierr                )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetValues( (HYPRE_StructMatrix) *matrix,
                                     (HYPRE_Int *)              grid_index,
                                     (HYPRE_Int)                *num_stencil_indices,
                                     (HYPRE_Int *)              stencil_indices,
                                     (double *)           values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetboxvalues, HYPRE_STRUCTMATRIXSETBOXVALUES)( hypre_F90_Obj *matrix,
                                                 HYPRE_Int      *ilower,
                                                 HYPRE_Int      *iupper,
                                                 HYPRE_Int      *num_stencil_indices,
                                                 HYPRE_Int      *stencil_indices,
                                                 double   *values,
                                                 HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (HYPRE_Int *)              ilower,
                                        (HYPRE_Int *)              iupper,
                                        (HYPRE_Int)                *num_stencil_indices,
                                        (HYPRE_Int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixgetboxvalues, HYPRE_STRUCTMATRIXGETBOXVALUES)( hypre_F90_Obj *matrix,
                                                 HYPRE_Int      *ilower,
                                                 HYPRE_Int      *iupper,
                                                 HYPRE_Int      *num_stencil_indices,
                                                 HYPRE_Int      *stencil_indices,
                                                 double   *values,
                                                 HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixGetBoxValues( (HYPRE_StructMatrix) *matrix,
                                        (HYPRE_Int *)              ilower,
                                        (HYPRE_Int *)              iupper,
                                        (HYPRE_Int)                *num_stencil_indices,
                                        (HYPRE_Int *)              stencil_indices,
                                        (double *)           values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetconstantva, HYPRE_STRUCTMATRIXSETCONSTANTVA)( hypre_F90_Obj *matrix,
                                                HYPRE_Int      *num_stencil_indices,
                                                HYPRE_Int      *stencil_indices,
                                                double   *values,
                                                HYPRE_Int      *ierr                )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetConstantValues( (HYPRE_StructMatrix) *matrix,
                                             (HYPRE_Int)       *num_stencil_indices,
                                             (HYPRE_Int *)     stencil_indices,
                                             (double *)  values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtovalues, HYPRE_STRUCTMATRIXADDTOVALUES)( hypre_F90_Obj *matrix,
                                                HYPRE_Int      *grid_index,
                                                HYPRE_Int      *num_stencil_indices,
                                                HYPRE_Int      *stencil_indices,
                                                double   *values,
                                                HYPRE_Int      *ierr                )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixAddToValues( (HYPRE_StructMatrix) *matrix,
                                       (HYPRE_Int *)     grid_index,
                                       (HYPRE_Int)       *num_stencil_indices,
                                       (HYPRE_Int *)     stencil_indices,
                                       (double *)  values           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtoboxvalues, HYPRE_STRUCTMATRIXADDTOBOXVALUES)( hypre_F90_Obj *matrix,
                                                   HYPRE_Int    *ilower,
                                                   HYPRE_Int    *iupper,
                                                   HYPRE_Int    *num_stencil_indices,
                                                   HYPRE_Int    *stencil_indices,
                                                   double *values,
                                                   HYPRE_Int    *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixAddToBoxValues( (HYPRE_StructMatrix) *matrix,
                                          (HYPRE_Int *)     ilower,
                                          (HYPRE_Int *)     iupper,
                                          (HYPRE_Int)       *num_stencil_indices,
                                          (HYPRE_Int *)     stencil_indices,
                                          (double *)  values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixaddtoconstant, HYPRE_STRUCTMATRIXADDTOCONSTANT)( hypre_F90_Obj *matrix,
                                                   HYPRE_Int    *num_stencil_indices,
                                                   HYPRE_Int    *stencil_indices,
                                                   double *values,
                                                   HYPRE_Int    *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetConstantValues( (HYPRE_StructMatrix) *matrix,
                                             (HYPRE_Int)       *num_stencil_indices,
                                             (HYPRE_Int *)     stencil_indices,
                                             (double *)  values        ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixassemble, HYPRE_STRUCTMATRIXASSEMBLE)( hypre_F90_Obj *matrix,
                                             HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixAssemble( (HYPRE_StructMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetnumghost, HYPRE_STRUCTMATRIXSETNUMGHOST)( hypre_F90_Obj *matrix,
                                                HYPRE_Int      *num_ghost,
                                                HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetNumGhost( (HYPRE_StructMatrix) *matrix,
                                       (HYPRE_Int *)               num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetgrid, HYPRE_STRUCTMATRIXGETGRID)( hypre_F90_Obj *matrix,
                                            hypre_F90_Obj *grid,
                                            HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixGetGrid( (HYPRE_StructMatrix) *matrix,
                                (HYPRE_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structmatrixsetsymmetric, HYPRE_STRUCTMATRIXSETSYMMETRIC)( hypre_F90_Obj *matrix,
                                                 HYPRE_Int      *symmetric,
                                                 HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetSymmetric( (HYPRE_StructMatrix) *matrix,
                                        (HYPRE_Int)                *symmetric ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantEntries
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixsetconstanten, HYPRE_STRUCTMATRIXSETCONSTANTEN)( hypre_F90_Obj *matrix,
                                                HYPRE_Int      *nentries,
                                                HYPRE_Int      *entries,
                                                HYPRE_Int      *ierr                )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixSetConstantEntries( (HYPRE_StructMatrix) *matrix,
                                              (HYPRE_Int)       *nentries,
                                              (HYPRE_Int *)     entries           ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixprint, HYPRE_STRUCTMATRIXPRINT)(
   hypre_F90_Obj *matrix,
   HYPRE_Int *all,
   HYPRE_Int *ierr )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructMatrixPrint("HYPRE_StructMatrix.out",
                                (HYPRE_StructMatrix) *matrix,
                                (HYPRE_Int)                *all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structmatrixmatvec, HYPRE_STRUCTMATRIXMATVEC)
                                              ( double   *alpha,
                                                hypre_F90_Obj *A,
                                                hypre_F90_Obj *x,
                                                double   *beta,
                                                hypre_F90_Obj *y,
                                                HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructMatrixMatvec( (double)             *alpha,
                                             (HYPRE_StructMatrix) *A,
                                             (HYPRE_StructVector) *x,
                                             (double)             *beta,
                                             (HYPRE_StructVector) *y  ) );
}

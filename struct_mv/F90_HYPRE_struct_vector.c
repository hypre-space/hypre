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
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorcreate, HYPRE_STRUCTVECTORCREATE)( hypre_F90_Comm *comm,
                                           hypre_F90_Obj *grid,
                                           hypre_F90_Obj *vector,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorCreate( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructVector *) vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectordestroy, HYPRE_STRUCTVECTORDESTROY)( hypre_F90_Obj *vector,
                                            HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructVectorDestroy( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorinitialize, HYPRE_STRUCTVECTORINITIALIZE)( hypre_F90_Obj *vector,
                                               HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorInitialize( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetvalues, HYPRE_STRUCTVECTORSETVALUES)( hypre_F90_Obj *vector,
                                              HYPRE_Int      *grid_index,
                                              double   *values,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorSetValues( (HYPRE_StructVector) *vector,
                                     (HYPRE_Int *)              grid_index,
                                     (double)             *values     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetboxvalues, HYPRE_STRUCTVECTORSETBOXVALUES)( hypre_F90_Obj *vector,
                                                 HYPRE_Int      *ilower,
                                                 HYPRE_Int      *iupper,
                                                 double   *values,
                                                 HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorSetBoxValues( (HYPRE_StructVector) *vector,
                                        (HYPRE_Int *)              ilower,
                                        (HYPRE_Int *)              iupper,
                                        (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtovalues, HYPRE_STRUCTVECTORADDTOVALUES)( hypre_F90_Obj *vector,
                                                HYPRE_Int      *grid_index,
                                                double   *values,
                                                HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorAddToValues( (HYPRE_StructVector) *vector,
                                       (HYPRE_Int *)              grid_index,
                                       (double)             *values     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtoboxvalue, HYPRE_STRUCTVECTORADDTOBOXVALUE)( hypre_F90_Obj *vector,
                                                   HYPRE_Int      *ilower,
                                                   HYPRE_Int      *iupper,
                                                   double   *values,
                                                   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorAddToBoxValues( (HYPRE_StructVector) *vector,
                                          (HYPRE_Int *)              ilower,
                                          (HYPRE_Int *)              iupper,
                                          (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorscalevalues, HYPRE_STRUCTVECTORSCALEVALUES)
                                             ( hypre_F90_Obj *vector,
                                               double   *factor,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorScaleValues( (HYPRE_StructVector) *vector,
                                       (double)             *factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetvalues, HYPRE_STRUCTVECTORGETVALUES)( hypre_F90_Obj *vector,
                                              HYPRE_Int      *grid_index,
                                              double   *values_ptr,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorGetValues( (HYPRE_StructVector) *vector,
                                     (HYPRE_Int *)              grid_index,
                                     (double *)           values_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetboxvalues, HYPRE_STRUCTVECTORGETBOXVALUES)( hypre_F90_Obj *vector,
                                                 HYPRE_Int      *ilower,
                                                 HYPRE_Int      *iupper,
                                                 double   *values,
                                                 HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorGetBoxValues( (HYPRE_StructVector) *vector,
                                        (HYPRE_Int *)              ilower,
                                        (HYPRE_Int *)              iupper,
                                        (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorassemble, HYPRE_STRUCTVECTORASSEMBLE)( hypre_F90_Obj *vector,
                                             HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorAssemble( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structvectorsetnumghost, HYPRE_STRUCTVECTORSETNUMGHOST)( hypre_F90_Obj *vector,
                                                HYPRE_Int      *num_ghost,
                                                HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorSetNumGhost( (HYPRE_StructVector) *vector,
                                       (HYPRE_Int *)              num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCopy
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structvectorcopy, HYPRE_STRUCTVECTORCOPY)
                                              ( hypre_F90_Obj *x,
                                                hypre_F90_Obj *y,
                                                HYPRE_Int      *ierr )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructVectorCopy( (HYPRE_StructVector) *x,
                                           (HYPRE_StructVector) *y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetconstantva, HYPRE_STRUCTVECTORSETCONSTANTVA)
                                               ( hypre_F90_Obj *vector,
                                                  double   *values,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorSetConstantValues( (HYPRE_StructVector)  *vector,
                                             (double)              *values ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetmigratecom, HYPRE_STRUCTVECTORGETMIGRATECOM)( hypre_F90_Obj *from_vector,
                                                  hypre_F90_Obj *to_vector,
                                                  hypre_F90_Obj *comm_pkg,
                                                  HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorGetMigrateCommPkg( (HYPRE_StructVector) *from_vector,
                                             (HYPRE_StructVector) *to_vector,
                                             (HYPRE_CommPkg *)    comm_pkg    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectormigrate, HYPRE_STRUCTVECTORMIGRATE)( hypre_F90_Obj *comm_pkg,
                                            hypre_F90_Obj *from_vector,
                                            hypre_F90_Obj *to_vector,
                                            HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorMigrate( (HYPRE_CommPkg)      *comm_pkg,
                                   (HYPRE_StructVector) *from_vector,
                                   (HYPRE_StructVector) *to_vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroycommpkg, HYPRE_DESTROYCOMMPKG)( hypre_F90_Obj *comm_pkg,
                                       HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_CommPkgDestroy( (HYPRE_CommPkg) *comm_pkg ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorprint, HYPRE_STRUCTVECTORPRINT)(
   hypre_F90_Obj *vector,
   HYPRE_Int       *all,
   HYPRE_Int       *ierr )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructVectorPrint("HYPRE_StructVector.out",
                                (HYPRE_StructVector) *vector,
                                (HYPRE_Int)                *all) );
}

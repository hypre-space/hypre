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
 * HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcreate, HYPRE_PARVECTORCREATE)( hypre_F90_Comm *comm,
                                     HYPRE_Int      *global_size,
                                     hypre_F90_Obj *partitioning,
                                     hypre_F90_Obj *vector,
                                     HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int)
             ( HYPRE_ParVectorCreate( (MPI_Comm) *comm,
                                      (HYPRE_Int)      *global_size,
                                      (HYPRE_Int *)    *partitioning,
                                      (HYPRE_ParVector *)  vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parmultivectorcreate, HYPRE_PARMULTIVECTORCREATE)
                                   ( hypre_F90_Comm *comm,
                                     HYPRE_Int      *global_size,
                                     hypre_F90_Obj *partitioning,
                                     HYPRE_Int      *number_vectors,
                                     hypre_F90_Obj *vector,
                                     HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int)
             ( HYPRE_ParMultiVectorCreate( (MPI_Comm) *comm,
                                      (HYPRE_Int)      *global_size,
                                      (HYPRE_Int *)    *partitioning,
                                      (HYPRE_Int)      *number_vectors,
                                      (HYPRE_ParVector *)  vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectordestroy, HYPRE_PARVECTORDESTROY)( hypre_F90_Obj *vector,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorDestroy( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinitialize, HYPRE_PARVECTORINITIALIZE)( hypre_F90_Obj *vector,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorInitialize( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectorread, HYPRE_PARVECTORREAD)( hypre_F90_Comm *comm,
                                      hypre_F90_Obj *vector,
                                      char     *file_name,
                                      HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorRead( (MPI_Comm) *comm,
                                        (char *)    file_name,
                                        (HYPRE_ParVector *) vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorprint, HYPRE_PARVECTORPRINT)( hypre_F90_Obj *vector,
                                       char     *fort_file_name,
                                       HYPRE_Int      *fort_file_name_size,
                                       HYPRE_Int      *ierr       )
{
   HYPRE_Int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
     c_file_name[i] = fort_file_name[i];

   *ierr = (HYPRE_Int) ( HYPRE_ParVectorPrint ( (HYPRE_ParVector) *vector,
                                          (char *)           c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetconstantvalue, HYPRE_PARVECTORSETCONSTANTVALUE)
                                          ( hypre_F90_Obj *vector,
                                            double   *value,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorSetConstantValues( (HYPRE_ParVector) *vector,
                                                     (double)          *value) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetrandomvalues, HYPRE_PARVECTORSETRANDOMVALUES)
                                          ( hypre_F90_Obj *vector,
                                            HYPRE_Int      *seed,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) *vector,
                                                   (HYPRE_Int)             *seed) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcopy, HYPRE_PARVECTORCOPY)
                                          ( hypre_F90_Obj *x,
                                            hypre_F90_Obj *y,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorCopy( (HYPRE_ParVector) *x,
                                        (HYPRE_ParVector) *y) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcloneshallow, HYPRE_PARVECTORCLONESHALLOW)
                                          ( hypre_F90_Obj *x,
                                            hypre_F90_Obj *xclone,
                                            HYPRE_Int     *ierr    )
{
   *xclone =
      (hypre_F90_Obj) ( HYPRE_ParVectorCloneShallow( (HYPRE_ParVector) *x ) );
   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorscale, HYPRE_PARVECTORSCALE)
                                          ( double   *value,
                                            hypre_F90_Obj *x,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorScale( (double)          *value,
                                         (HYPRE_ParVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectoraxpy, HYPRE_PARVECTORAXPY)
                                          ( double   *value,
                                            hypre_F90_Obj *x,
                                            hypre_F90_Obj *y,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorAxpy( (double)          *value,
                                        (HYPRE_ParVector) *x,
                                        (HYPRE_ParVector) *y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinnerprod, HYPRE_PARVECTORINNERPROD)
                                           (hypre_F90_Obj *x,
                                            hypre_F90_Obj *y,
                                            double   *prod,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParVectorInnerProd( (HYPRE_ParVector) *x,
                                             (HYPRE_ParVector) *y,
                                             (double *)         prod ) );
}

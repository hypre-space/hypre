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
 * HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorcreate, HYPRE_IJVECTORCREATE)(
                                                    HYPRE_Int      *comm,
                                                    HYPRE_Int      *jlower,
                                                    HYPRE_Int      *jupper,
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorCreate( (MPI_Comm)         *comm,
                                         (HYPRE_Int)              *jlower,
                                         (HYPRE_Int)              *jupper, 
                                         (HYPRE_IJVector *)  vector  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectordestroy, HYPRE_IJVECTORDESTROY)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorDestroy( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorinitialize, HYPRE_IJVECTORINITIALIZE)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorInitialize( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorsetvalues, HYPRE_IJVECTORSETVALUES)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *num_values,
                                                    HYPRE_Int      *indices,
                                                    double   *values,
                                                    HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorSetValues( (HYPRE_IJVector) *vector,
                                            (HYPRE_Int)            *num_values,
                                            (const HYPRE_Int *)     indices,
                                            (const double *)  values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectoraddtovalues, HYPRE_IJVECTORADDTOVALUES)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *num_values,
                                                    HYPRE_Int      *indices,
                                                    double   *values,
                                                    HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorAddToValues( (HYPRE_IJVector) *vector,
                                              (HYPRE_Int)            *num_values,
                                              (const HYPRE_Int *)     indices,
                                              (const double *)  values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorassemble, HYPRE_IJVECTORASSEMBLE)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorAssemble( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetvalues, HYPRE_IJVECTORGETVALUES)(
                                                    hypre_F90_Obj *vector,
                                                    const HYPRE_Int *num_values,
                                                    const HYPRE_Int *indices,
                                                    double   *values,
                                                    HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorGetValues( (HYPRE_IJVector) *vector,
                                            (HYPRE_Int)            *num_values,
                                            (const HYPRE_Int *)     indices,
                                            (double *)        values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetmaxoffprocelmt, HYPRE_IJVECTORSETMAXOFFPROCELMT)
                                              ( hypre_F90_Obj *vector,
                                                HYPRE_Int       *max_off_proc_elmts,
                                                HYPRE_Int       *ierr    )
{
   *ierr = (HYPRE_Int) 
          ( HYPRE_IJVectorSetMaxOffProcElmts( (HYPRE_IJVector) *vector,
                                              (HYPRE_Int)            *max_off_proc_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetobjecttype, HYPRE_IJVECTORSETOBJECTTYPE)(
                                                    hypre_F90_Obj *vector,
                                                    const HYPRE_Int *type,
                                                    HYPRE_Int       *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJVectorSetObjectType( (HYPRE_IJVector) *vector,
                                                (HYPRE_Int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobjecttype, HYPRE_IJVECTORGETOBJECTTYPE)(
                                                    hypre_F90_Obj *vector,
                                                    HYPRE_Int      *type,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) (HYPRE_IJVectorGetObjectType( (HYPRE_IJVector) *vector,
                                               (HYPRE_Int *)           type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalRange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalrange, HYPRE_IJVECTORGETLOCALRANGE)
                                           ( hypre_F90_Obj *vector,
                                             HYPRE_Int      *jlower,
                                             HYPRE_Int      *jupper,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) (HYPRE_IJVectorGetLocalRange( (HYPRE_IJVector) *vector,
                                                    (HYPRE_Int *)           jlower,
                                                    (HYPRE_Int *)           jupper  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobject, HYPRE_IJVECTORGETOBJECT)(
                                                    hypre_F90_Obj *vector,
                                                    hypre_F90_Obj *object,
                                                    HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) (HYPRE_IJVectorGetObject( (HYPRE_IJVector) *vector,
                                                (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorread, HYPRE_IJVECTORREAD)( char     *filename,
                                                         HYPRE_Int *comm,
                                                         HYPRE_Int      *object_type,
                                                         hypre_F90_Obj *vector,
                                                         HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) (HYPRE_IJVectorRead( (char *)            filename,
                                      (MPI_Comm)         *comm,
                                      (HYPRE_Int)              *object_type,
                                      (HYPRE_IJVector *)  vector       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorprint, HYPRE_IJVECTORPRINT)( hypre_F90_Obj *vector,
                                                           char     *filename,
                                                           HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) (HYPRE_IJVectorPrint( (HYPRE_IJVector) *vector,
                                       (char *)          filename ) );
}

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
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixcreate, HYPRE_IJMATRIXCREATE)
                                    ( hypre_F90_Comm *comm,
                                      HYPRE_Int      *ilower,
                                      HYPRE_Int      *iupper,
                                      HYPRE_Int      *jlower,
                                      HYPRE_Int      *jupper,
                                      hypre_F90_Obj *matrix,
                                      HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixCreate( (MPI_Comm)         *comm,
                                         (HYPRE_Int)              *ilower,
                                         (HYPRE_Int)              *iupper,
                                         (HYPRE_Int)              *jlower,
                                         (HYPRE_Int)              *jupper,
                                         (HYPRE_IJMatrix *)  matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixdestroy, HYPRE_IJMATRIXDESTROY)
                                     ( hypre_F90_Obj *matrix,
                                       HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixDestroy( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinitialize, HYPRE_IJMATRIXINITIALIZE)
                                        ( hypre_F90_Obj *matrix,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixInitialize( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetvalues, HYPRE_IJMATRIXSETVALUES)(
                                                     hypre_F90_Obj *matrix,
                                                     HYPRE_Int          *nrows,
                                                     HYPRE_Int          *ncols,
                                                     const HYPRE_Int    *rows,
                                                     const HYPRE_Int    *cols,
                                                     const double *values,
                                                     HYPRE_Int          *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixSetValues( (HYPRE_IJMatrix) *matrix,
                                            (HYPRE_Int)            *nrows,
                                            (HYPRE_Int *)           ncols,
                                            (const HYPRE_Int *)     rows,
                                            (const HYPRE_Int *)     cols,
                                            (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtovalues, HYPRE_IJMATRIXADDTOVALUES)(
                                                     hypre_F90_Obj *matrix,
                                                     HYPRE_Int          *nrows,
                                                     HYPRE_Int          *ncols,
                                                     const HYPRE_Int    *rows,
                                                     const HYPRE_Int    *cols,
                                                     const double *values,
                                                     HYPRE_Int          *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixAddToValues( (HYPRE_IJMatrix) *matrix,
                                              (HYPRE_Int)            *nrows,
                                              (HYPRE_Int *)           ncols,
                                              (const HYPRE_Int *)     rows,
                                              (const HYPRE_Int *)     cols,
                                              (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixassemble, HYPRE_IJMATRIXASSEMBLE)
                                      ( hypre_F90_Obj *matrix,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixAssemble( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowCounts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixgetrowcounts, HYPRE_IJMATRIXGETROWCOUNTS)
                                          ( hypre_F90_Obj *matrix,
                                            HYPRE_Int       *nrows,
                                            HYPRE_Int       *rows,
                                            HYPRE_Int       *ncols,
                                            HYPRE_Int       *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixGetRowCounts((HYPRE_IJMatrix) *matrix,
                                              (HYPRE_Int)            *nrows,
                                              (HYPRE_Int *)           rows,
                                              (HYPRE_Int *)           ncols ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixgetvalues, HYPRE_IJMATRIXGETVALUES)
                                       ( hypre_F90_Obj *matrix,
                                         HYPRE_Int          *nrows,
                                         HYPRE_Int          *ncols,
                                         HYPRE_Int    *rows,
                                         HYPRE_Int    *cols,
                                         double *values,
                                         HYPRE_Int          *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixGetValues( (HYPRE_IJMatrix) *matrix,
                                            (HYPRE_Int)            *nrows,
                                            (HYPRE_Int *)           ncols,
                                            (HYPRE_Int *)           rows,
                                            (HYPRE_Int *)           cols,
                                            (double *)        values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobjecttype, HYPRE_IJMATRIXSETOBJECTTYPE)(
                                                     hypre_F90_Obj *matrix,
                                                     const HYPRE_Int *type,
                                                     HYPRE_Int       *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixSetObjectType( (HYPRE_IJMatrix) *matrix,
                                                (HYPRE_Int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobjecttype, HYPRE_IJMATRIXGETOBJECTTYPE)(
                                                     hypre_F90_Obj *matrix,
                                                     HYPRE_Int      *type,
                                                     HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixGetObjectType( (HYPRE_IJMatrix) *matrix,
                                                (HYPRE_Int *)           type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetlocalrange, HYPRE_IJMATRIXGETLOCALRANGE)
                                           ( hypre_F90_Obj *matrix,
                                             HYPRE_Int      *ilower,
                                             HYPRE_Int      *iupper,
                                             HYPRE_Int      *jlower,
                                             HYPRE_Int      *jupper,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixGetLocalRange( (HYPRE_IJMatrix) *matrix,
                                                (HYPRE_Int *)           ilower,
                                                (HYPRE_Int *)           iupper,
                                                (HYPRE_Int *)           jlower,
                                                (HYPRE_Int *)           jupper ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobject, HYPRE_IJMATRIXGETOBJECT)(
                                                     hypre_F90_Obj *matrix,
                                                     hypre_F90_Obj *object,
                                                     HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixGetObject( (HYPRE_IJMatrix) *matrix,
                                            (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetrowsizes, HYPRE_IJMATRIXSETROWSIZES)
                                         ( hypre_F90_Obj *matrix,
                                           const HYPRE_Int *sizes,
                                           HYPRE_Int       *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixSetRowSizes( (HYPRE_IJMatrix) *matrix,
                                              (const HYPRE_Int *)     sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetdiagoffdsizes, HYPRE_IJMATRIXSETDIAGOFFDSIZES)
                                              ( hypre_F90_Obj *matrix,
                                                const HYPRE_Int *diag_sizes,
                                                const HYPRE_Int *offd_sizes,
                                                HYPRE_Int       *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixSetDiagOffdSizes( (HYPRE_IJMatrix) *matrix,
                                                   (const HYPRE_Int *)     diag_sizes,
                                                   (const HYPRE_Int *)     offd_sizes ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetmaxoffprocelmt, HYPRE_IJMATRIXSETMAXOFFPROCELMT)
                                              ( hypre_F90_Obj *matrix,
                                                HYPRE_Int       *max_off_proc_elmts,
                                                HYPRE_Int       *ierr        )
{
   *ierr = (HYPRE_Int) 
         ( HYPRE_IJMatrixSetMaxOffProcElmts( (HYPRE_IJMatrix) *matrix,
                                             (HYPRE_Int)            *max_off_proc_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixread, HYPRE_IJMATRIXREAD)(
                                                     char     *filename,
                                                     hypre_F90_Comm *comm,
                                                     HYPRE_Int      *object_type,
                                                     hypre_F90_Obj *matrix,
                                                     HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixRead( (char *)            filename,
                                       (MPI_Comm)         *comm,
                                       (HYPRE_Int)              *object_type,
                                       (HYPRE_IJMatrix *)  matrix    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixprint, HYPRE_IJMATRIXPRINT)(
                                                     hypre_F90_Obj *matrix,
                                                     char     *filename,
                                                     HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_IJMatrixPrint( (HYPRE_IJMatrix) *matrix,
                                        (char *)          filename ) );
}

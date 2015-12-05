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
 * $Revision: 2.9 $
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
                                    ( long int *comm,
                                      int      *ilower,
                                      int      *iupper,
                                      int      *jlower,
                                      int      *jupper,
                                      long int *matrix,
                                      int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixCreate( (MPI_Comm)         *comm,
                                         (int)              *ilower,
                                         (int)              *iupper,
                                         (int)              *jlower,
                                         (int)              *jupper,
                                         (HYPRE_IJMatrix *)  matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixdestroy, HYPRE_IJMATRIXDESTROY)
                                     ( long int *matrix,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixDestroy( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixinitialize, HYPRE_IJMATRIXINITIALIZE)
                                        ( long int *matrix,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixInitialize( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetvalues, HYPRE_IJMATRIXSETVALUES)(
                                                     long int     *matrix,
                                                     int          *nrows,
                                                     int          *ncols,
                                                     const int    *rows,
                                                     const int    *cols,
                                                     const double *values,
                                                     int          *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetValues( (HYPRE_IJMatrix) *matrix,
                                            (int)            *nrows,
                                            (int *)           ncols,
                                            (const int *)     rows,
                                            (const int *)     cols,
                                            (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixaddtovalues, HYPRE_IJMATRIXADDTOVALUES)(
                                                     long int     *matrix,
                                                     int          *nrows,
                                                     int          *ncols,
                                                     const int    *rows,
                                                     const int    *cols,
                                                     const double *values,
                                                     int          *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAddToValues( (HYPRE_IJMatrix) *matrix,
                                              (int)            *nrows,
                                              (int *)           ncols,
                                              (const int *)     rows,
                                              (const int *)     cols,
                                              (const double *)  values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixassemble, HYPRE_IJMATRIXASSEMBLE)
                                      ( long int *matrix,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixAssemble( (HYPRE_IJMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowCounts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixgetrowcounts, HYPRE_IJMATRIXGETROWCOUNTS)
                                          ( long int  *matrix,
                                            int       *nrows,
                                            int       *rows,
                                            int       *ncols,
                                            int       *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixGetRowCounts((HYPRE_IJMatrix) *matrix,
                                              (int)            *nrows,
                                              (int *)           rows,
                                              (int *)           ncols ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixgetvalues, HYPRE_IJMATRIXGETVALUES)
                                       ( long int     *matrix,
                                         int          *nrows,
                                         int          *ncols,
                                         int    *rows,
                                         int    *cols,
                                         double *values,
                                         int          *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixGetValues( (HYPRE_IJMatrix) *matrix,
                                            (int)            *nrows,
                                            (int *)           ncols,
                                            (int *)           rows,
                                            (int *)           cols,
                                            (double *)        values  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobjecttype, HYPRE_IJMATRIXSETOBJECTTYPE)(
                                                     long int  *matrix,
                                                     const int *type,
                                                     int       *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetObjectType( (HYPRE_IJMatrix) *matrix,
                                                (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobjecttype, HYPRE_IJMATRIXGETOBJECTTYPE)(
                                                     long int *matrix,
                                                     int      *type,
                                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixGetObjectType( (HYPRE_IJMatrix) *matrix,
                                                (int *)           type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetlocalrange, HYPRE_IJMATRIXGETLOCALRANGE)
                                           ( long int *matrix,
                                             int      *ilower,
                                             int      *iupper,
                                             int      *jlower,
                                             int      *jupper,
                                             int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixGetLocalRange( (HYPRE_IJMatrix) *matrix,
                                                (int *)           ilower,
                                                (int *)           iupper,
                                                (int *)           jlower,
                                                (int *)           jupper ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixgetobject, HYPRE_IJMATRIXGETOBJECT)(
                                                     long int *matrix,
                                                     long int *object,
                                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixGetObject( (HYPRE_IJMatrix) *matrix,
                                            (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetrowsizes, HYPRE_IJMATRIXSETROWSIZES)
                                         ( long int  *matrix,
                                           const int *sizes,
                                           int       *ierr    )
{
   *ierr = (int) ( HYPRE_IJMatrixSetRowSizes( (HYPRE_IJMatrix) *matrix,
                                              (const int *)     sizes   ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetdiagoffdsizes, HYPRE_IJMATRIXSETDIAGOFFDSIZES)
                                              ( long int  *matrix,
                                                const int *diag_sizes,
                                                const int *offd_sizes,
                                                int       *ierr        )
{
   *ierr = (int) ( HYPRE_IJMatrixSetDiagOffdSizes( (HYPRE_IJMatrix) *matrix,
                                                   (const int *)     diag_sizes,
                                                   (const int *)     offd_sizes ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_ijmatrixsetmaxoffprocelmt, HYPRE_IJMATRIXSETMAXOFFPROCELMT)
                                              ( long int  *matrix,
                                                int       *max_off_proc_elmts,
                                                int       *ierr        )
{
   *ierr = (int) 
         ( HYPRE_IJMatrixSetMaxOffProcElmts( (HYPRE_IJMatrix) *matrix,
                                             (int)            *max_off_proc_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixread, HYPRE_IJMATRIXREAD)(
                                                     char     *filename,
                                                     long int *comm,
                                                     int      *object_type,
                                                     long int *matrix,
                                                     int      *ierr      )
{
   *ierr = (int) ( HYPRE_IJMatrixRead( (char *)            filename,
                                       (MPI_Comm)         *comm,
                                       (int)              *object_type,
                                       (HYPRE_IJMatrix *)  matrix    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixprint, HYPRE_IJMATRIXPRINT)(
                                                     long int *matrix,
                                                     char     *filename,
                                                     int      *ierr      )
{
   *ierr = (int) ( HYPRE_IJMatrixPrint( (HYPRE_IJMatrix) *matrix,
                                        (char *)          filename ) );
}

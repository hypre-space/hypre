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
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcreate, HYPRE_SSTRUCTVECTORCREATE)
                                                              (int      *comm,
                                                               long int *grid,
                                                               long int *vector_ptr,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorCreate( (MPI_Comm)             *comm,
                                             (HYPRE_SStructGrid)    *grid,
                                             (HYPRE_SStructVector *) vector_ptr ) );
}

/*--------------------------------------------------------------------------
HYPRE_SStructVectorDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectordestroy, HYPRE_SSTRUCTVECTORDESTROY)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorDestroy( (HYPRE_SStructVector) *vector ) );
}

/*---------------------------------------------------------
HYPRE_SStructVectorInitialize
 * ----------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorinitialize, HYPRE_SSTRUCTVECTORINITIALIZE)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorInitialize( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetvalues, HYPRE_SSTRUCTVECTORSETVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetValues( (HYPRE_SStructVector) *vector,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetboxvalues, HYPRE_SSTRUCTVECTORSETBOXVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetBoxValues( (HYPRE_SStructVector) *vector,
                                                   (int)                 *part,
                                                   (int *)                ilower,
                                                   (int *)                iupper,
                                                   (int)                 *var,
                                                   (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtovalues, HYPRE_SSTRUCTVECTORADDTOVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAddToValues( (HYPRE_SStructVector) *vector,
                                                  (int)                 *part,
                                                  (int *)                index,
                                                  (int)                 *var,
                                                  (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtoboxvalu, HYPRE_SSTRUCTVECTORADDTOBOXVALU)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAddToBoxValues( (HYPRE_SStructVector) *vector,
                                                     (int)                 *part,
                                                     (int *)                ilower,
                                                     (int *)                iupper,
                                                     (int)                 *var,
                                                     (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorassemble, HYPRE_SSTRUCTVECTORASSEMBLE)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAssemble( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgather, HYPRE_SSTRUCTVECTORGATHER)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGather( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetvalues, HYPRE_SSTRUCTVECTORGETVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetValues( (HYPRE_SStructVector) *vector,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (double *)             value ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetboxvalues, HYPRE_SSTRUCTVECTORGETBOXVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetBoxValues( (HYPRE_SStructVector ) *vector,
                                                   (int)                  *part,
                                                   (int *)                 ilower,
                                                   (int *)                 iupper,
                                                   (int)                  *var,
                                                   (double *)              values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetconstantv, HYPRE_SSTRUCTVECTORSETCONSTANTV)
                                                              (long int *vector,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetConstantValues( (HYPRE_SStructVector) *vector,
                                                        (double)              *value));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetobjecttyp, HYPRE_SSTRUCTVECTORSETOBJECTTYP)
                                                              (long int *vector,
                                                               int      *type,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetObjectType( (HYPRE_SStructVector) *vector,
                                                    (int)                 *type ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetobject, HYPRE_SSTRUCTVECTORGETOBJECT)
                                                              (long int *vector,
                                                               long int *object,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetObject( (HYPRE_SStructVector) *vector,
                                                (void **)              object ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorprint, HYPRE_SSTRUCTVECTORPRINT)
                                                              (const char *filename,
                                                               long int   *vector,
                                                               int        *all,
                                                               int        *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorPrint( (const char * )        filename,
                                            (HYPRE_SStructVector) *vector,
                                            (int)                 *all ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcopy, HYPRE_SSTRUCTVECTORCOPY)
                                                         (long int   *x,
                                                          long int   *y,
                                                          int        *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorCopy( (HYPRE_SStructVector) *x,
                                           (HYPRE_SStructVector) *y ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorscale, HYPRE_SSTRUCTVECTORSCALE)
                                                         (double   *alpha,
                                                          long int *y,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorScale( (double)              *alpha,
                                            (HYPRE_SStructVector) *y ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructinnerprod, HYPRE_SSTRUCTINNERPROD)
                                                         (long int   *x,
                                                          long int   *y,
                                                          double     *result,
                                                          int        *ierr)
{
   *ierr = (int) (HYPRE_SStructInnerProd( (HYPRE_SStructVector) *x,
                                          (HYPRE_SStructVector) *y,
                                          (double *)             result ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructaxpy, HYPRE_SSTRUCTAXPY)
                                                 (double   *alpha,
                                                  long int *x,
                                                  long int *y,
                                                  int      *ierr)
{
   *ierr = (int) (HYPRE_SStructAxpy( (double)              *alpha,
                                     (HYPRE_SStructVector) *x,
                                     (HYPRE_SStructVector) *y ) );
}

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
 * HYPRE_DistributedMatrix interface
 *
 *****************************************************************************/

#include "./distributed_matrix.h"


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixCreate( MPI_Comm context, HYPRE_DistributedMatrix *matrix )
{
   int ierr = 0;

   *matrix = (HYPRE_DistributedMatrix)
	    hypre_DistributedMatrixCreate( context );

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixDestroy( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixLimitedDestroy( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixLimitedDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixInitialize( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixInitialize( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixAssemble( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAssemble( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixSetLocalStorageType( HYPRE_DistributedMatrix matrix,
				 int               type           )
{
   return( hypre_DistributedMatrixSetLocalStorageType(
      (hypre_DistributedMatrix *) matrix, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixGetLocalStorageType( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetLocalStorageType(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixSetLocalStorage( HYPRE_DistributedMatrix matrix,
				      void                 *LocalStorage )
{
   return( hypre_DistributedMatrixSetLocalStorage(
      (hypre_DistributedMatrix *) matrix, LocalStorage ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
HYPRE_DistributedMatrixGetLocalStorage( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetLocalStorage(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetTranslator
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixSetTranslator( HYPRE_DistributedMatrix matrix,
				      void                 *Translator )
{
   return( hypre_DistributedMatrixSetTranslator(
      (hypre_DistributedMatrix *) matrix, Translator ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetTranslator
 *--------------------------------------------------------------------------*/

void *
HYPRE_DistributedMatrixGetTranslator( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetTranslator(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetAuxiliaryData
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixSetAuxiliaryData( HYPRE_DistributedMatrix matrix,
				      void                 *AuxiliaryData )
{
   return( hypre_DistributedMatrixSetAuxiliaryData(
      (hypre_DistributedMatrix *) matrix, AuxiliaryData ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
HYPRE_DistributedMatrixGetAuxiliaryData( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAuxiliaryData(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetContext
 *--------------------------------------------------------------------------*/

MPI_Comm
HYPRE_DistributedMatrixGetContext( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixContext(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetDims
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixGetDims( HYPRE_DistributedMatrix matrix, 
                               int *M, int *N )
{
   int ierr=0;

   *M = hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix );
   *N = hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetDims
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixSetDims( HYPRE_DistributedMatrix matrix, 
                               int M, int N )
{
   int ierr=0;

   hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix ) = M;
   hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix ) = N;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPrint
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixPrint( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixPrint( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

int
HYPRE_DistributedMatrixGetLocalRange( HYPRE_DistributedMatrix matrix, 
                               int *row_start, int *row_end ,
                               int *col_start, int *col_end )
{
   return( hypre_DistributedMatrixGetLocalRange( (hypre_DistributedMatrix *) matrix,
                             row_start, row_end, col_start, col_end ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixGetRow( HYPRE_DistributedMatrix matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   return( hypre_DistributedMatrixGetRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixRestoreRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_DistributedMatrixRestoreRow( HYPRE_DistributedMatrix matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   return( hypre_DistributedMatrixRestoreRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

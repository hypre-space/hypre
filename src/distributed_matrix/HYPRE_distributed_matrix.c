/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_DistributedMatrix interface
 *
 *****************************************************************************/

#include "./distributed_matrix.h"


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixCreate( MPI_Comm context, HYPRE_DistributedMatrix *matrix )
{
   HYPRE_Int ierr = 0;

   *matrix = (HYPRE_DistributedMatrix)
	    hypre_DistributedMatrixCreate( context );

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixDestroy( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixLimitedDestroy( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixLimitedDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixInitialize( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixInitialize( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixAssemble( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAssemble( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_DistributedMatrixSetLocalStorageType( HYPRE_DistributedMatrix matrix,
				 HYPRE_Int               type           )
{
   return( hypre_DistributedMatrixSetLocalStorageType(
      (hypre_DistributedMatrix *) matrix, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_DistributedMatrixGetLocalStorageType( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetLocalStorageType(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
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

HYPRE_Int
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

HYPRE_Int
HYPRE_DistributedMatrixGetDims( HYPRE_DistributedMatrix matrix, 
                               HYPRE_BigInt *M, HYPRE_BigInt *N )
{
   HYPRE_Int ierr=0;

   *M = hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix );
   *N = hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixSetDims
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_DistributedMatrixSetDims( HYPRE_DistributedMatrix matrix, 
                               HYPRE_BigInt M, HYPRE_BigInt N )
{
   HYPRE_Int ierr=0;

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

HYPRE_Int 
HYPRE_DistributedMatrixPrint( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixPrint( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_DistributedMatrixGetLocalRange( HYPRE_DistributedMatrix matrix, 
                               HYPRE_BigInt *row_start, HYPRE_BigInt *row_end ,
                               HYPRE_BigInt *col_start, HYPRE_BigInt *col_end )
{
   return( hypre_DistributedMatrixGetLocalRange( (hypre_DistributedMatrix *) matrix,
                             row_start, row_end, col_start, col_end ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_DistributedMatrixGetRow( HYPRE_DistributedMatrix matrix,
                             HYPRE_BigInt row,
                             HYPRE_Int *size,
                             HYPRE_BigInt **col_ind,
                             HYPRE_Real **values )
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

HYPRE_Int 
HYPRE_DistributedMatrixRestoreRow( HYPRE_DistributedMatrix matrix,
                             HYPRE_BigInt row,
                             HYPRE_Int *size,
                             HYPRE_BigInt **col_ind,
                             HYPRE_Real **values )
{
   return( hypre_DistributedMatrixRestoreRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

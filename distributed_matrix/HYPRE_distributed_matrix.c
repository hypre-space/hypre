/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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

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
 * HYPRE_NewDistributedMatrix
 *--------------------------------------------------------------------------*/

HYPRE_DistributedMatrix 
HYPRE_NewDistributedMatrix( MPI_Comm    context )
{
   return ( (HYPRE_DistributedMatrix)
	    hypre_NewDistributedMatrix( context ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeDistributedMatrix( HYPRE_DistributedMatrix matrix )
{
   return( hypre_FreeDistributedMatrix( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_InitializeDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_InitializeDistributedMatrix( HYPRE_DistributedMatrix matrix )
{
   return( hypre_InitializeDistributedMatrix( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleDistributedMatrix( HYPRE_DistributedMatrix matrix )
{
   return( hypre_AssembleDistributedMatrix( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetDistributedMatrixLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_SetDistributedMatrixLocalStorageType( HYPRE_DistributedMatrix matrix,
				 int               type           )
{
   return( hypre_SetDistributedMatrixLocalStorageType(
      (hypre_DistributedMatrix *) matrix, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixLocalStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_GetDistributedMatrixLocalStorageType( HYPRE_DistributedMatrix matrix )
{
   return( hypre_GetDistributedMatrixLocalStorageType(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetDistributedMatrixLocalStorage
 *--------------------------------------------------------------------------*/

int
HYPRE_SetDistributedMatrixLocalStorage( HYPRE_DistributedMatrix matrix,
				      void                 *LocalStorage )
{
   return( hypre_SetDistributedMatrixLocalStorage(
      (hypre_DistributedMatrix *) matrix, LocalStorage ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixLocalStorage
 *--------------------------------------------------------------------------*/

void *
HYPRE_GetDistributedMatrixLocalStorage( HYPRE_DistributedMatrix matrix )
{
   return( hypre_GetDistributedMatrixLocalStorage(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetDistributedMatrixTranslator
 *--------------------------------------------------------------------------*/

int
HYPRE_SetDistributedMatrixTranslator( HYPRE_DistributedMatrix matrix,
				      void                 *Translator )
{
   return( hypre_SetDistributedMatrixTranslator(
      (hypre_DistributedMatrix *) matrix, Translator ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixTranslator
 *--------------------------------------------------------------------------*/

void *
HYPRE_GetDistributedMatrixTranslator( HYPRE_DistributedMatrix matrix )
{
   return( hypre_GetDistributedMatrixTranslator(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetDistributedMatrixAuxiliaryData
 *--------------------------------------------------------------------------*/

int
HYPRE_SetDistributedMatrixAuxiliaryData( HYPRE_DistributedMatrix matrix,
				      void                 *AuxiliaryData )
{
   return( hypre_SetDistributedMatrixAuxiliaryData(
      (hypre_DistributedMatrix *) matrix, AuxiliaryData ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
HYPRE_GetDistributedMatrixAuxiliaryData( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAuxiliaryData(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixContext
 *--------------------------------------------------------------------------*/

MPI_Comm
HYPRE_GetDistributedMatrixContext( HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixContext(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixDims
 *--------------------------------------------------------------------------*/

int
HYPRE_GetDistributedMatrixDims( HYPRE_DistributedMatrix matrix, 
                               int *M, int *N )
{
   int ierr=0;

   *M = hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix );
   *N = hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetDistributedMatrixDims
 *--------------------------------------------------------------------------*/

int
HYPRE_SetDistributedMatrixDims( HYPRE_DistributedMatrix matrix, 
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
 * HYPRE_PrintDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_PrintDistributedMatrix( HYPRE_DistributedMatrix matrix )
{
   return( hypre_PrintDistributedMatrix( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixLocalRange
 *--------------------------------------------------------------------------*/

int
HYPRE_GetDistributedMatrixLocalRange( HYPRE_DistributedMatrix matrix, 
                               int *start, int *end )
{
   return( hypre_GetDistributedMatrixLocalRange( (hypre_DistributedMatrix *) matrix,
                             start, end ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDistributedMatrixRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetDistributedMatrixRow( HYPRE_DistributedMatrix matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   return( hypre_GetDistributedMatrixRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_RestoreDistributedMatrixRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_RestoreDistributedMatrixRow( HYPRE_DistributedMatrix matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   return( hypre_RestoreDistributedMatrixRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

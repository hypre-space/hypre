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
 * Member functions for hypre_DistributedMatrix class.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

/*--------------------------------------------------------------------------
 *     BASIC CONSTRUCTION/DESTRUCTION SEQUENCE
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_NewDistributedMatrix
 *--------------------------------------------------------------------------*/

hypre_DistributedMatrix *
hypre_NewDistributedMatrix( MPI_Comm     context  )
{
   hypre_DistributedMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_DistributedMatrix, 1);

   hypre_DistributedMatrixContext(matrix) = context;
   hypre_DistributedMatrixM(matrix)    = -1;
   hypre_DistributedMatrixN(matrix)    = -1;
   hypre_DistributedMatrixAuxiliaryData(matrix)    = NULL;
   hypre_DistributedMatrixAuxiliaryData(matrix)    = NULL;
   hypre_DistributedMatrixLocalStorage(matrix) = NULL;
   hypre_DistributedMatrixTranslator(matrix) = NULL;
   hypre_DistributedMatrixLocalStorageType(matrix) = HYPRE_UNITIALIZED;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_FreeDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeDistributedMatrix( hypre_DistributedMatrix *matrix )
{

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      hypre_FreeDistributedMatrixPETSc( matrix );
   else
      return(-1);

   hypre_TFree(matrix);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_LimitedFreeDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_LimitedFreeDistributedMatrix( hypre_DistributedMatrix *matrix )
{

   hypre_TFree(matrix);

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_InitializeDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeDistributedMatrix( hypre_DistributedMatrix *matrix )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( 0 );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_AssembleDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleDistributedMatrix( hypre_DistributedMatrix *matrix )
{
   if( (hypre_DistributedMatrixLocalStorageType(matrix) != HYPRE_PETSC_MATRIX )
     )
     return(-1);

   if( hypre_DistributedMatrixLocalStorage(matrix) == NULL )
     return(-1);

   if( (hypre_DistributedMatrixM(matrix) < 0 ) ||
       (hypre_DistributedMatrixN(matrix) < 0 ) )
     return(-1);

   return(0);
}

/*--------------------------------------------------------------------------
 *     Get/Sets that are independent of underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_SetDistributedMatrixLocalStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_SetDistributedMatrixLocalStorageType( hypre_DistributedMatrix *matrix,
				 int                type   )
{
   hypre_DistributedMatrixLocalStorageType(matrix) = type;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalStorageType( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixLocalStorageType(matrix) );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetDistributedMatrixLocalStorage
 *--------------------------------------------------------------------------*/

int 
hypre_SetDistributedMatrixLocalStorage( hypre_DistributedMatrix *matrix,
				 void                  *local_storage  )
{
   hypre_DistributedMatrixLocalStorage(matrix) = local_storage;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalStorage
 *--------------------------------------------------------------------------*/

void *
hypre_GetDistributedMatrixLocalStorage( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixLocalStorage(matrix) );

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_SetDistributedMatrixTranslator
 *--------------------------------------------------------------------------*/

int 
hypre_SetDistributedMatrixTranslator( hypre_DistributedMatrix *matrix,
				 void                  *translator  )
{
   hypre_DistributedMatrixTranslator(matrix) = translator;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixTranslator
 *--------------------------------------------------------------------------*/

void *
hypre_GetDistributedMatrixTranslator( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixTranslator(matrix) );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetDistributedMatrixAuxiliaryData
 *--------------------------------------------------------------------------*/

int 
hypre_SetDistributedMatrixAuxiliaryData( hypre_DistributedMatrix *matrix,
				 void                  *auxiliary_data  )
{
   hypre_DistributedMatrixAuxiliaryData(matrix) = auxiliary_data;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
hypre_GetDistributedMatrixAuxiliaryData( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixAuxiliaryData(matrix) );

   return(0);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_PrintDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_PrintDistributedMatrix( hypre_DistributedMatrix *matrix )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_PrintDistributedMatrixPETSc( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalRange
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalRange( hypre_DistributedMatrix *matrix,
                             int *start,
                             int *end )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_GetDistributedMatrixLocalRangePETSc( matrix,
                             start,
                             end ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixRow
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixRow( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_GetDistributedMatrixRowPETSc( matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_RestoreDistributedMatrixRow
 *--------------------------------------------------------------------------*/

int 
hypre_RestoreDistributedMatrixRow( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_RestoreDistributedMatrixRowPETSc( matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
   else
      return(-1);
}

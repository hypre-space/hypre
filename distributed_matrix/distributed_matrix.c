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
#include "../HYPRE.h"

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

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      hypre_FreeDistributedMatrixPETSc( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      hypre_FreeDistributedMatrixISIS( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      hypre_FreeDistributedMatrixParcsr( matrix );
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
   int ierr = 0;

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( 0 );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_InitializeDistributedMatrixISIS(matrix);
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_InitializeDistributedMatrixParcsr(matrix);
   else
      ierr = -1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleDistributedMatrix( hypre_DistributedMatrix *matrix )
{

   if( 
       (hypre_DistributedMatrixLocalStorageType(matrix) != HYPRE_PETSC )
    && (hypre_DistributedMatrixLocalStorageType(matrix) != HYPRE_ISIS )
    && (hypre_DistributedMatrixLocalStorageType(matrix) != HYPRE_PARCSR )
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
   int ierr=0;

   hypre_DistributedMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalStorageType( hypre_DistributedMatrix *matrix  )
{
   int ierr=0;

   ierr = hypre_DistributedMatrixLocalStorageType(matrix);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetDistributedMatrixLocalStorage
 *--------------------------------------------------------------------------*/

int 
hypre_SetDistributedMatrixLocalStorage( hypre_DistributedMatrix *matrix,
				 void                  *local_storage  )
{
   int ierr=0;

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
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_PrintDistributedMatrixPETSc( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_PrintDistributedMatrixISIS( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_PrintDistributedMatrixParcsr( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalRange
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalRange( hypre_DistributedMatrix *matrix,
                             int *row_start,
                             int *row_end,
                             int *col_start,
                             int *col_end )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_GetDistributedMatrixLocalRangePETSc( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_GetDistributedMatrixLocalRangeISIS( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_GetDistributedMatrixLocalRangeParcsr( matrix, row_start, row_end, col_start, col_end ) );
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
   int ierr = 0;

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC ) {
      ierr = hypre_GetDistributedMatrixRowPETSc( matrix, row, size, col_ind, values );
      return( ierr );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS ) {
      ierr = hypre_GetDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
      return( ierr );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR ) {
      ierr = hypre_GetDistributedMatrixRowParcsr( matrix, row, size, col_ind, values );
      return( ierr );
   }
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
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_RestoreDistributedMatrixRowPETSc( matrix, row, size, col_ind, values ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_RestoreDistributedMatrixRowISIS( matrix, row, size, col_ind, values ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_RestoreDistributedMatrixRowParcsr( matrix, row, size, col_ind, values ) );
   else
      return(-1);
}

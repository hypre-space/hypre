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
 * hypre_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_DistributedMatrix *
hypre_DistributedMatrixCreate( MPI_Comm     context  )
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
 * hypre_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixDestroy( hypre_DistributedMatrix *matrix )
{

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      hypre_DistributedMatrixDestroyPETSc( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      hypre_FreeDistributedMatrixISIS( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      hypre_DistributedMatrixDestroyParCSR( matrix );
   else
      return(-1);

   hypre_TFree(matrix);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixLimitedDestroy( hypre_DistributedMatrix *matrix )
{

   hypre_TFree(matrix);

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixInitialize( hypre_DistributedMatrix *matrix )
{
   int ierr = 0;

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( 0 );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_InitializeDistributedMatrixISIS(matrix);
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_DistributedMatrixInitializeParCSR(matrix);
   else
      ierr = -1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixAssemble( hypre_DistributedMatrix *matrix )
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
 * hypre_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixSetLocalStorageType( hypre_DistributedMatrix *matrix,
				 int                type   )
{
   int ierr=0;

   hypre_DistributedMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetLocalStorageType( hypre_DistributedMatrix *matrix  )
{
   int ierr=0;

   ierr = hypre_DistributedMatrixLocalStorageType(matrix);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixSetLocalStorage( hypre_DistributedMatrix *matrix,
				 void                  *local_storage  )
{
   int ierr=0;

   hypre_DistributedMatrixLocalStorage(matrix) = local_storage;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetLocalStorage( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixLocalStorage(matrix) );

}


/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetTranslator
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixSetTranslator( hypre_DistributedMatrix *matrix,
				 void                  *translator  )
{
   hypre_DistributedMatrixTranslator(matrix) = translator;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetTranslator
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetTranslator( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixTranslator(matrix) );

}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetAuxiliaryData
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixSetAuxiliaryData( hypre_DistributedMatrix *matrix,
				 void                  *auxiliary_data  )
{
   hypre_DistributedMatrixAuxiliaryData(matrix) = auxiliary_data;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetAuxiliaryData( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixAuxiliaryData(matrix) );

}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrint
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixPrint( hypre_DistributedMatrix *matrix )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_DistributedMatrixPrintPETSc( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_PrintDistributedMatrixISIS( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_DistributedMatrixPrintParCSR( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetLocalRange( hypre_DistributedMatrix *matrix,
                             int *row_start,
                             int *row_end,
                             int *col_start,
                             int *col_end )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_DistributedMatrixGetLocalRangePETSc( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_GetDistributedMatrixLocalRangeISIS( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_DistributedMatrixGetLocalRangeParCSR( matrix, row_start, row_end, col_start, col_end ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetRow( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr = 0;

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC ) {
      ierr = hypre_DistributedMatrixGetRowPETSc( matrix, row, size, col_ind, values );
      return( ierr );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS ) {
      ierr = hypre_GetDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
      return( ierr );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR ) {
      ierr = hypre_DistributedMatrixGetRowParCSR( matrix, row, size, col_ind, values );
      return( ierr );
   }
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRow
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixRestoreRow( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      return( hypre_DistributedMatrixRestoreRowPETSc( matrix, row, size, col_ind, values ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      return( hypre_RestoreDistributedMatrixRowISIS( matrix, row, size, col_ind, values ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      return( hypre_DistributedMatrixRestoreRowParCSR( matrix, row, size, col_ind, values ) );
   else
      return(-1);
}

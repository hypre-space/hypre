/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_matrix_vector.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

int HYPRE_IJMatrixCreate( MPI_Comm comm, HYPRE_IJMatrix *in_matrix_ptr, 
          int global_m, int global_n)

{
   int ierr=0;

   hypre_IJMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_IJMatrix, 1);

   hypre_IJMatrixContext(matrix) = comm;
   hypre_IJMatrixM(matrix)    = global_m;
   hypre_IJMatrixN(matrix)    = global_n;
   hypre_IJMatrixLocalStorage(matrix) = NULL;
   hypre_IJMatrixTranslator(matrix) = NULL;
   hypre_IJMatrixLocalStorageType(matrix) = HYPRE_UNITIALIZED;
   hypre_IJMatrixInsertionSemantics(matrix) = 0;
   hypre_IJMatrixReferenceCount(matrix) = 1;

   *in_matrix_ptr = (HYPRE_IJMatrix) matrix;
  
   return( ierr ); 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDestroy( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if (matrix)
   {
      hypre_IJMatrixReferenceCount( matrix ) --;
   
      if ( hypre_IJMatrixReferenceCount( matrix ) <= 0 )
      {
	/*
         if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
            ierr = hypre_IJMatrixDestroyPETSc( matrix );
         else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
            ierr = hypre_IJMatrixDestroyISIS( matrix );
         else */ 
	 if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
            ierr = hypre_IJMatrixDestroyParCSR( matrix );
         else
            ierr = -1;

         hypre_TFree(matrix);
      }
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixInitialize( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInitializePETSc( matrix );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInitializeISIS( matrix );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInitializeParCSR( matrix );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAssemble( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if( (hypre_IJMatrixM(matrix) < 0 ) ||
       (hypre_IJMatrixN(matrix) < 0 ) )
      return(-1);

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAssemblePETSc( matrix );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAssembleISIS( matrix );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixAssembleParCSR( matrix );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDistribute
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDistribute( HYPRE_IJMatrix IJmatrix, 
			  const int     *row_starts,
			  const int     *col_starts )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixDistributeParCSR( matrix, row_starts, col_starts );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetLocalStorageType( HYPRE_IJMatrix IJmatrix, int type )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   hypre_IJMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetLocalSize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetLocalSize( HYPRE_IJMatrix IJmatrix, int local_m, int local_n )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetLocalSizePETSc (matrix, local_m, local_n);
   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetLocalSizeISIS (matrix, local_m, local_n);
      */
   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetLocalSizeParCSR (matrix, local_m, local_n);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagRowSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetDiagRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetDiagRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetDiagRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetOffDiagRowSizes
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetOffDiagRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetOffDiagRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetOffDiagRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixQueryInsertionSemantics
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixQueryInsertionSemantics( HYPRE_IJMatrix IJmatrix, int *level )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   *level = hypre_IJMatrixInsertionSemantics(matrix);

   return(ierr);

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertBlock
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixInsertBlock( HYPRE_IJMatrix IJmatrix, int m, int n,
                           const int *rows, const int *cols, 
			   const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /*  if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInsertBlockPETSc( matrix, m, n, rows, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInsertBlockISIS( matrix, m, n, rows, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInsertBlockParCSR( matrix, m, n, rows, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToBlock
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixAddToBlock( HYPRE_IJMatrix IJmatrix, int m, int n,
                          const int *rows, const int *cols, 
			  const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToBlockPETSc( matrix, m, n, rows, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToBlockISIS( matrix, m, n, rows, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixAddToBlockParCSR( matrix, m, n, rows, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertRow
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInsertRowPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInsertRowISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixInsertRowParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRow
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixAddToRow( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToRowPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToRowISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixAddToRowParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRowAfter
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixAddToRowAfter( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToRowafterPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToRowAfterISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixAddToRowAfterParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetValues( HYPRE_IJMatrix IJmatrix,
                         int            n,
                         int            row,
                         const int     *cols, 
                         const double  *values )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
   {
      ierr = hypre_IJMatrixSetValuesPETSc( matrix, n, row, cols, values, 0 );
   }
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
   {
      ierr = hypre_IJMatrixSetValuesISIS( matrix, n, row, cols, values, 0 );
   }
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
   {
      ierr = hypre_IJMatrixSetValuesParCSR( matrix, n, row, cols, values, 0 );
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix IJmatrix,
                           int            n,
                           int            row,
                           const int     *cols, 
                           const double  *values )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
   {
      ierr = hypre_IJMatrixSetValuesPETSc( matrix, n, row, cols, values, 1 );
   }
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
   {
      ierr = hypre_IJMatrixSetValuesISIS( matrix, n, row, cols, values, 1 );
   }
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
   {
      ierr = hypre_IJMatrixSetValuesParCSR( matrix, n, row, cols, values, 1 );
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetBlockValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetBlockValues( HYPRE_IJMatrix IJmatrix,
                              int            m,
                              int            n,
                              const int     *rows,
                              const int     *cols, 
                              const double  *values )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
   {
      ierr = hypre_IJMatrixSetBlockValuesPETSc( matrix, m, n,
                                                rows, cols, values, 0 );
   }
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
   {
      ierr = hypre_IJMatrixSetBlockValuesISIS( matrix, m, n,
                                               rows, cols, values, 0 );
   }
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
   {
      ierr = hypre_IJMatrixSetBlockValuesParCSR( matrix, m, n,
                                                 rows, cols, values, 0 );
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToBlockValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAddToBlockValues( HYPRE_IJMatrix IJmatrix,
                                int            m,
                                int            n,
                                const int     *rows,
                                const int     *cols, 
                                const double  *values )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
   {
      ierr = hypre_IJMatrixSetBlockValuesPETSc( matrix, m, n,
                                                rows, cols, values, 1 );
   }
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
   {
      ierr = hypre_IJMatrixSetBlockValuesISIS( matrix, m, n,
                                               rows, cols, values, 1 );
   }
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
   {
      ierr = hypre_IJMatrixSetBlockValuesParCSR( matrix, m, n,
                                                 rows, cols, values, 1 );
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/***************************************************************************
 * The following are routines that are not generally used by or supported
 * for users
 ***************************************************************************/


/*--------------------------------------------------------------------------
 * hypre_RefIJMatrix
 *--------------------------------------------------------------------------*/

/**
Sets a reference to point to an IJMatrix.

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
@param reference [OUT]
The pointer to be set to point to IJMatrix.
*/

int 
hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix, HYPRE_IJMatrix *reference )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   hypre_IJMatrixReferenceCount(matrix) ++;

   *reference = IJmatrix;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to an underlying matrix type used to implement IJMatrix.
Assumes that the implementation has an underlying matrix, so it would not
work with a direct implementation of IJMatrix. 

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

void *
HYPRE_IJMatrixGetLocalStorage( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   return( hypre_IJMatrixLocalStorage( matrix ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning if IJmatrix has an underlying
parcsr matrix

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

int
HYPRE_IJMatrixGetRowPartitioning( HYPRE_IJMatrix IJmatrix ,
				  const int    **row_partitioning )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetRowPartitioningParCSR( matrix ,
                                                     row_partitioning );
   else
      ierr = -1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning if IJmatrix has an underlying
parcsr matrix

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

int
HYPRE_IJMatrixGetColPartitioning( HYPRE_IJMatrix IJmatrix ,
				  const int    **col_partitioning )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetColPartitioningParCSR( matrix ,
                                                     col_partitioning );
   else
      ierr = -1;

   return ierr;
}

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
 * HYPRE_CSRMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_CSRMatrixCreate( int  num_rows,
                       int  num_cols,
                       int *row_sizes )
{
   hypre_CSRMatrix *matrix;
   int             *matrix_i;
   int              i;

   matrix_i = hypre_CTAlloc(int, num_rows + 1);
   matrix_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      matrix_i[i+1] = matrix_i[i] + row_sizes[i];
   }

   matrix = hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;

   return ( (HYPRE_CSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix )
{
   return( hypre_CSRMatrixDestroy( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix )
{
   return ( hypre_CSRMatrixInitialize( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixRead
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_CSRMatrixRead( char            *file_name )
{
   return ( (HYPRE_CSRMatrix) hypre_CSRMatrixRead( file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

void 
HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix  matrix,
                      char            *file_name )
{
   hypre_CSRMatrixPrint( (hypre_CSRMatrix *) matrix,
                         file_name );
}


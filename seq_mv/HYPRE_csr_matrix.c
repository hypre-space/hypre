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
 * HYPRE_CreateCSRMatrix
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_CreateCSRMatrix( int  num_rows,
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

   matrix = hypre_CreateCSRMatrix(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;

   return ( (HYPRE_CSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_DestroyCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return( hypre_DestroyCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeCSRMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return ( hypre_InitializeCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintCSRMatrix
 *--------------------------------------------------------------------------*/

void 
HYPRE_PrintCSRMatrix( HYPRE_CSRMatrix  matrix,
                      char            *file_name )
{
   hypre_PrintCSRMatrix( (hypre_CSRMatrix *) matrix,
                         file_name );
}


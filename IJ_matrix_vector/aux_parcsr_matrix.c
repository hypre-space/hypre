
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
 * Member functions for hypre_AuxParCSRMatrix class.
 *
 *****************************************************************************/

#include "IJ_matrix_vector.h"
#include "aux_parcsr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_CreateAuxParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_AuxParCSRMatrix *
hypre_CreateAuxParCSRMatrix( int  local_num_rows,
                       	     int  local_num_cols,
			     int *sizes)
{
   hypre_AuxParCSRMatrix  *matrix;
   int *row_space;
   int i;
   
   matrix = hypre_CTAlloc(hypre_AuxParCSRMatrix, 1);
  
   hypre_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   hypre_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;

   if (sizes)
   {
      row_space = hypre_CTAlloc (int, local_num_rows);
      for (i=0 ; i < local_num_rows; i++)
      {
	 row_space[i] = sizes[i];
      }
      hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
   }
   else
   {
      hypre_AuxParCSRMatrixRowSpace(matrix) = NULL;
   }

   /* set defaults */
   hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   hypre_AuxParCSRMatrixRowLength(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxData(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxOffd(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_DestroyAuxParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_DestroyAuxParCSRMatrix( hypre_AuxParCSRMatrix *matrix )
{
   int ierr=0;
   int i;
   int num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);

   if (matrix)
   {
      if (hypre_AuxParCSRMatrixRowLength(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowLength(matrix));
      if (hypre_AuxParCSRMatrixRowSpace(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowSpace(matrix));
      if (hypre_AuxParCSRMatrixAuxJ(matrix))
      {
         for (i=0; i < num_rows; i++)
	    hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[i]);
	 hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix));
      }
      if (hypre_AuxParCSRMatrixAuxData(matrix))
      {
         for (i=0; i < num_rows; i++)
            hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[i]);
	 hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix));
      }
      if (hypre_AuxParCSRMatrixIndxDiag(matrix))
            hypre_TFree(hypre_AuxParCSRMatrixIndxDiag(matrix));
      if (hypre_AuxParCSRMatrixIndxOffd(matrix))
            hypre_TFree(hypre_AuxParCSRMatrixIndxOffd(matrix));
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeAuxParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeAuxParCSRMatrix( hypre_AuxParCSRMatrix *matrix )
{
   int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   int local_num_cols = hypre_AuxParCSRMatrixLocalNumCols(matrix);
   int *row_space = hypre_AuxParCSRMatrixRowSpace(matrix);
   int **aux_j;
   double **aux_data;
   int i;

   if (local_num_rows <= 0) 
      return -1;
   if (hypre_AuxParCSRMatrixNeedAux(matrix))
   {
      aux_j = hypre_CTAlloc(int *, local_num_rows);
      aux_data = hypre_CTAlloc(double *, local_num_rows);
      if (!hypre_AuxParCSRMatrixRowLength(matrix))
         hypre_AuxParCSRMatrixRowLength(matrix) = 
  		hypre_CTAlloc(int, local_num_rows);
      if (row_space)
      {
         for (i=0; i < local_num_rows; i++)
         {
            aux_j[i] = hypre_CTAlloc(int, row_space[i]);
            aux_data[i] = hypre_CTAlloc(double, row_space[i]);
         }
      }
      else
      {
         row_space = hypre_CTAlloc(int, local_num_rows);
         for (i=0; i < local_num_rows; i++)
         {
            row_space[i] = 30;
            aux_j[i] = hypre_CTAlloc(int, 30);
            aux_data[i] = hypre_CTAlloc(double, 30);
         }
         hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
      }
      hypre_AuxParCSRMatrixAuxJ(matrix) = aux_j;
      hypre_AuxParCSRMatrixAuxData(matrix) = aux_data;
   }
   else
   {
      hypre_AuxParCSRMatrixIndxDiag(matrix) = hypre_CTAlloc(int,local_num_rows);
      hypre_AuxParCSRMatrixIndxOffd(matrix) = hypre_CTAlloc(int,local_num_rows);
   }

   return 0;
}

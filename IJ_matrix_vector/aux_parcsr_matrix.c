
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
 * hypre_CreateParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_AuxParCSRMatrix *
hypre_CreateAuxParCSRMatrix( int local_num_rows,
                       	     int local_num_cols,
			     int diag_size,
			     int offd_size)
{
   hypre_AuxParCSRMatrix  *matrix;
   int indx_diag, indx_offd;
   
   matrix = hypre_CTAlloc(hypre_AuxParCSRMatrix, 1);
  
   hypre_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   hypre_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;
   hypre_AuxParCSRMatrixDiagSize(matrix) = diag_size;
   hypre_AuxParCSRMatrixOffdSize(matrix) = offd_size;

   /* set defaults */
   hypre_AuxParCSRMatrixIndxDiag(matrix) = 0;
   hypre_AuxParCSRMatrixIndxOffd(matrix) = 0;
   hypre_AuxParCSRMatrixRowStartDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixRowEndDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixRowStartOffd(matrix) = NULL;
   hypre_AuxParCSRMatrixRowEndOffd(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxDiagJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxDiagData(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxOffdJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxOffdData(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_DestroyAuxParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_DestroyAuxParCSRMatrix( hypre_AuxParCSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      if (hypre_AuxParCSRMatrixRowStartDiag(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowStartDiag(matrix));
      if (hypre_AuxParCSRMatrixRowEndDiag(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowEndDiag(matrix));
      if (hypre_AuxParCSRMatrixDiagSize(matrix) > 0)
      {
         hypre_TFree(hypre_AuxParCSRMatrixAuxDiagJ(matrix));
         hypre_TFree(hypre_AuxParCSRMatrixAuxDiagData(matrix));
      }
      if (hypre_AuxParCSRMatrixRowStartOffd(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowStartOffd(matrix));
      if (hypre_AuxParCSRMatrixRowStartOffd(matrix))
         hypre_TFree(hypre_AuxParCSRMatrixRowEndOffd(matrix));
      if (hypre_AuxParCSRMatrixOffdSize(matrix) > 0)
      {
         hypre_TFree(hypre_AuxParCSRMatrixAuxOffdJ(matrix));
         hypre_TFree(hypre_AuxParCSRMatrixAuxOffdData(matrix));
      }
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
   int  ierr=0;
   int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   int local_num_cols = hypre_AuxParCSRMatrixLocalNumCols(matrix);
   int diag_size = hypre_AuxParCSRMatrixDiagSize(matrix);
   int offd_size = hypre_AuxParCSRMatrixOffdSize(matrix);
   if (diag_size != -2)
   {
      hypre_AuxParCSRMatrixRowStartDiag(matrix) = 
  		hypre_CTAlloc(int, local_num_rows+1);
      hypre_AuxParCSRMatrixRowEndDiag(matrix) = 
 		hypre_CTAlloc(int, local_num_rows+1);
   }
   if (diag_size > 0)
   {
      hypre_AuxParCSRMatrixAuxDiagJ(matrix) = hypre_CTAlloc(int, diag_size);
      hypre_AuxParCSRMatrixAuxDiagData(matrix) = hypre_CTAlloc(double, diag_size);
   }
   if (offd_size != -2)
   {
      hypre_AuxParCSRMatrixRowStartOffd(matrix) = 
 		hypre_CTAlloc(int, local_num_rows+1);
      hypre_AuxParCSRMatrixRowEndOffd(matrix) = 
 		hypre_CTAlloc(int, local_num_rows+1);
   }
   if (offd_size > 0)
   {
      hypre_AuxParCSRMatrixAuxOffdJ(matrix) = hypre_CTAlloc(int, offd_size);
      hypre_AuxParCSRMatrixAuxOffdData(matrix) = hypre_CTAlloc(double, offd_size);
   }
   return ierr;
}



/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Finds transpose of a hypre_CSRMatrix
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/


int hypre_CSRMatrixTranspose(hypre_CSRMatrix   *A, hypre_CSRMatrix   **AT)

{
   double       *A_data = hypre_CSRMatrixData(A);
   int          *A_i = hypre_CSRMatrixI(A);
   int          *A_j = hypre_CSRMatrixJ(A);
   int           num_rowsA = hypre_CSRMatrixNumRows(A);
   int           num_colsA = hypre_CSRMatrixNumCols(A);
   int           num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);

   double       *AT_data;
   int          *AT_i;
   int          *AT_j;
   int           num_rowsAT;
   int           num_colsAT;
   int           num_nonzerosAT;

   int           max_col;
   int           i, j;

   /*-------------------------------------------------------------- 
    * First, ascertain that num_cols and num_nonzeros has been set. 
    * If not, set them.
    *--------------------------------------------------------------*/

   if (! num_nonzerosA)
   {
      num_nonzerosA = A_i[num_rowsA];
   }

   if (! num_colsA)
   {
      max_col = 0;
      for (i = 0; i < num_rowsA; ++i)
      {
          for (j = A_i[i]; j < A_i[i+1]; j++)
          {
              if (A_j[j] > max_col)
                 max_col = A_j[j];
          }
      }
      num_colsA = max_col+1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = hypre_CSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT);
   hypre_CSRMatrixInitialize(*AT);

   AT_data = hypre_CSRMatrixData(*AT);
   AT_i = hypre_CSRMatrixI(*AT);
   AT_j = hypre_CSRMatrixJ(*AT);

   /*-----------------------------------------------------------------
    * Count the number of entries in each column of A (row of AT)
    * and fill the AT_i array.
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_nonzerosA; i++)
   {
       ++AT_i[A_j[i]+1];
   }

   for (i = 2; i <= num_rowsAT; i++)
   {
       AT_i[i] += AT_i[i-1];
   }

   /*----------------------------------------------------------------
    * Load the data and column numbers of AT
    *----------------------------------------------------------------*/

   for (i = 0; i < num_rowsA; i++)
   {
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         AT_j[AT_i[A_j[j]]] = i;
         AT_data[AT_i[A_j[j]]] = A_data[j];
         AT_i[A_j[j]]++;
      }
   }

   /*------------------------------------------------------------
    * AT_i[j] now points to the *end* of the jth row of entries
    * instead of the beginning.  Restore AT_i to front of row.
    *------------------------------------------------------------*/

   for (i = num_rowsAT; i > 0; i--)
   {
         AT_i[i] = AT_i[i-1];
   }

   AT_i[0] = 0;

   return(0);
}




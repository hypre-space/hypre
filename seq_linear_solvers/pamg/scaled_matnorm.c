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
 * computes |D^-1/2 A D^-1/2 |_sup where D diagonal matrix 
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixScaledNorm
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixScaledNorm( hypre_CSRMatrix *A, double *scnorm)
{
   int			*diag_i = hypre_CSRMatrixI(A);
   int			*diag_j = hypre_CSRMatrixJ(A);
   double		*diag_data = hypre_CSRMatrixData(A);
   int			 num_rows = hypre_CSRMatrixNumRows(A);

   hypre_Vector         *dinvsqrt;
   double		*dis_data;
   hypre_Vector         *sum;
   double		*sum_data;
  
   int	      i, j;

   double      mat_norm;

   dinvsqrt = hypre_VectorCreate(num_rows);
   hypre_VectorInitialize(dinvsqrt);
   dis_data = hypre_VectorData(dinvsqrt);
   sum = hypre_VectorCreate(num_rows);
   hypre_VectorInitialize(sum);
   sum_data = hypre_VectorData(sum);

   /* generate dinvsqrt */
   for (i=0; i < num_rows; i++)
   {
      dis_data[i] = 1.0/sqrt(fabs(diag_data[diag_i[i]]));
   }
   
   for (i=0; i < num_rows; i++)
   {
      for (j=diag_i[i]; j < diag_i[i+1]; j++)
      {
	 sum_data[i] += fabs(diag_data[j])*dis_data[i]*dis_data[diag_j[j]];
      }
   }   

   mat_norm = 0;
   for (i=0; i < num_rows; i++)
   {
      if (mat_norm < sum_data[i]) 
	 mat_norm = sum_data[i];
   }	

   hypre_VectorDestroy(dinvsqrt);
   hypre_VectorDestroy(sum);

   *scnorm = mat_norm;  
   return 0;
}

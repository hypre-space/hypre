/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * computes |D^-1/2 A D^-1/2 |_sup where D diagonal matrix 
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixScaledNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixScaledNorm( hypre_CSRMatrix *A, double *scnorm)
{
   HYPRE_Int			*diag_i = hypre_CSRMatrixI(A);
   HYPRE_Int			*diag_j = hypre_CSRMatrixJ(A);
   double		*diag_data = hypre_CSRMatrixData(A);
   HYPRE_Int			 num_rows = hypre_CSRMatrixNumRows(A);

   hypre_Vector         *dinvsqrt;
   double		*dis_data;
   hypre_Vector         *sum;
   double		*sum_data;
  
   HYPRE_Int	      i, j;

   double      mat_norm;

   dinvsqrt = hypre_SeqVectorCreate(num_rows);
   hypre_SeqVectorInitialize(dinvsqrt);
   dis_data = hypre_VectorData(dinvsqrt);
   sum = hypre_SeqVectorCreate(num_rows);
   hypre_SeqVectorInitialize(sum);
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

   hypre_SeqVectorDestroy(dinvsqrt);
   hypre_SeqVectorDestroy(sum);

   *scnorm = mat_norm;  
   return 0;
}

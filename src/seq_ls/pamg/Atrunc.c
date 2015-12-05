/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/






#include "headers.h"

         
HYPRE_Int
hypre_AMGOpTruncation(hypre_CSRMatrix *A, double trunc_factor, HYPRE_Int max_elmts)
{
   HYPRE_Int ierr = 0;
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   double *A_data = hypre_CSRMatrixData(A);
   double max_coef, row_sum, scale;
   HYPRE_Int i, j, start, diag;
   HYPRE_Int num_variables = hypre_CSRMatrixNumRows(A);
   HYPRE_Int now_checking, num_lost, next_open;

   if (trunc_factor > 0)
   {
      num_lost = 0;
      next_open = 0;
      now_checking = 0;
      for (i=0; i < num_variables; i++)
      {
         max_coef = 0;
	 if (max_coef < fabs(A_data[A_i[i]])) max_coef = fabs(A_data[A_i[i]]);
         max_coef *= trunc_factor;
      }
      for (i=0; i < num_variables; i++)
      {
         start = A_i[i];
         A_i[i] -= num_lost;
	 diag = next_open;
	 A_data[next_open] = A_data[now_checking];
	 A_j[next_open] = A_j[now_checking];
	 now_checking++;
	 next_open++;
         for (j = start+1; j < A_i[i+1]; j++)
         {
	    if (fabs(A_data[now_checking]) < max_coef)
	    {
	       num_lost++;
	       now_checking++;
	       A_data[diag] += A_data[now_checking];
	    }
	    else
	    {
	       A_data[next_open] = A_data[now_checking];
	       A_j[next_open] = A_j[now_checking];
	       now_checking++;
	       next_open++;
	    }
         }
      }
      A_i[num_variables] -= num_lost;
      hypre_CSRMatrixNumNonzeros(A) = A_i[num_variables];
   
   }
   else if (max_elmts > 0)
   {
      next_open = 0;
      start = 0;
      for (i=0; i < num_variables; i++)
      {
         row_sum = 0;
         for (j=A_i[i]; j < A_i[i+1]; j++)
	    row_sum += A_data[j];
	 start = next_open;
	 if (A_i[i] > next_open)
	 {
	    for (j = A_i[i]; j < A_i[i+1]; j++)
	    {
	       A_data[next_open] = A_data[j];
	       A_j[next_open++] = A_j[j];
	    }
	 }
 	 if ((A_i[i+1]-A_i[i]) > max_elmts)
	 {
	    qsort2(A_j, A_data, start, next_open-1);
	    next_open = start+max_elmts;
	 }
	 else
	 {
	    next_open = start + (A_i[i+1]-A_i[i]);
	 }

         scale = 0;
	 for (j=start; j < next_open; j++)
 	 {
	    scale += A_data[j];
 	 }
         
         if (scale != 0 && scale != row_sum)
         {
	    scale = row_sum/scale;
            for (j=start; j < next_open; j++)
	       A_data[j] *= scale;
         }
	 A_i[i] = start;
      }
      A_i[num_variables] = next_open;
      hypre_CSRMatrixNumNonzeros(A) = next_open;
   }

   return ierr;
}

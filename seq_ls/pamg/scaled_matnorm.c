/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

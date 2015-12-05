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





#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGBuildInterpCR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGBuildCRInterp( hypre_CSRMatrix  *A,
                   HYPRE_Int                 *CF_marker,
		   HYPRE_Int			n_coarse,
		   HYPRE_Int			num_relax_steps,
		   HYPRE_Int			relax_type,
		   double	        relax_weight,
                   hypre_CSRMatrix     **P_ptr )
{
   
   HYPRE_Int             *A_i;
   HYPRE_Int             *A_j;

   hypre_CSRMatrix *P; 
   hypre_Vector	   *zero_vector;
   hypre_Vector	   *x_vector;
   hypre_Vector	   *tmp_vector;
   double          *x_data;

   double          *P_data;
   HYPRE_Int             *P_i;
   HYPRE_Int             *P_j;

   HYPRE_Int              P_size;
   
   HYPRE_Int             *P_marker;

   HYPRE_Int              n_fine;

   HYPRE_Int             *coarse_to_fine;
   HYPRE_Int              coarse_counter;
   
   HYPRE_Int              i,ic,i1,i2;
   HYPRE_Int              j,jj;
   HYPRE_Int              kk,k1;
   HYPRE_Int              extended_nghbr;
   
   double           summ, sump;
   
   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   n_fine = hypre_CSRMatrixNumRows(A);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   extended_nghbr = 0;
   if (num_relax_steps > 1) extended_nghbr = 1;
   coarse_counter = 0;

   coarse_to_fine = hypre_CTAlloc(HYPRE_Int, n_coarse);

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   P_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);

   for (i = 0; i < n_fine; i++)
   {
      
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] > 0)
      {
         coarse_to_fine[coarse_counter] = i;
         coarse_counter++;
      
	 i2 = i+2;
	 P_marker[i] = i2;
	 P_i[i+1]++;          
         for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
         {
            i1 = A_j[jj]; 
	    if (CF_marker[i1] < 0)
	    {
	       if (P_marker[i1] != i2) 
	       {
		  P_i[i1+1]++;          
	          P_marker[i1] = i2;
	       }
	       if (extended_nghbr)
	       {
                  for (kk = A_i[i1]+1; kk < A_i[i1+1]; kk++)
                  {
	             k1 = A_j[kk];
	             if (CF_marker[k1] < 0)
	             {
	                if (P_marker[k1] != i2) 
                        {
			   P_i[k1+1]++;	
                           P_marker[k1] = i2;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   for (i = 1; i < n_fine; i++)
      P_i[i+1] += P_i[i];

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_size = P_i[n_fine];

   P_j    = hypre_CTAlloc(HYPRE_Int, P_size);
   P_data = hypre_CTAlloc(double, P_size);
   zero_vector = hypre_SeqVectorCreate(n_fine);
   x_vector = hypre_SeqVectorCreate(n_fine);
   tmp_vector = hypre_SeqVectorCreate(n_fine);
   hypre_SeqVectorInitialize(zero_vector);
   hypre_SeqVectorInitialize(x_vector);
   hypre_SeqVectorInitialize(tmp_vector);
   x_data = hypre_VectorData(x_vector);

   /*-----------------------------------------------------------------------
    *  Second Pass: Define interpolation and fill in P_data, P_i, and P_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   for (ic = 0; ic < n_coarse; ic++)
   {
      i = coarse_to_fine[ic];
      i2 = i+2;
      P_marker[i] = 0;
      for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
      {
         i1 = A_j[jj]; 
         if (CF_marker[i1] < 0) 
	 {
	    P_marker[i1] = i2;
	    if (extended_nghbr)
	    {
               for (kk = A_i[i1]+1; kk < A_i[i1+1]; kk++)
               {
	          k1 = A_j[kk];
                  if (CF_marker[k1] < 0) 
			P_marker[k1] = i2;
                  else
			P_marker[k1] = 0;
               }
            }
         }
      }
      hypre_SeqVectorSetConstantValues(x_vector, 0.0);
      x_data[i] = 1.0;
      for (jj = 0; jj < num_relax_steps; jj++) 
         hypre_AMGRelax(A, zero_vector, P_marker, relax_type, i2,
			   relax_weight, x_vector, tmp_vector);
      for (jj = 0; jj < n_fine; jj++)
      {
	 if (P_marker[jj] == i2)
	 {
	    P_j[P_i[jj]] = ic;
	    P_data[P_i[jj]] = x_data[jj];
	    P_i[jj]++;
	 }
      }
      P_data[P_i[i]] = 1.0;
      P_j[P_i[i]] = ic;
      P_i[i]++;
   }
   for (i = n_fine-1; i > -1; i--)
      P_i[i+1] = P_i[i];
   P_i[0] = 0;
   
   /*--------------------------------------------------------------------
    *  global compatible relaxation
    *--------------------------------------------------------------------*/

   hypre_SeqVectorSetConstantValues(x_vector, 1.0);
   for (jj = 0; jj < num_relax_steps; jj++) 
       hypre_AMGRelax(A, zero_vector, CF_marker, relax_type, -1,
		      relax_weight, x_vector, tmp_vector);
  
   /*-----------------------------------------------------------------------
    *  perform normalization
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i  < n_fine  ; i ++)
   {
      if (CF_marker[i] < 0)
      {
	 sump = 0.0;
	 summ = 0.0;
	 for (j = P_i[i]; j < P_i[i+1]; j++)
	    if (P_data[j] > 0)
	       sump += P_data[j];
	    else
	       summ += P_data[j];
	 if (sump != 0) sump = x_data[i]/sump;
	 if (summ != 0) summ = x_data[i]/summ;
	 for (j = P_i[i]; j < P_i[i+1]; j++)
	    if (P_data[j] > 0)
               P_data[j] = P_data[j]*sump;
	    else
               P_data[j] = P_data[j]*summ;
      }
   }   
      
   P = hypre_CSRMatrixCreate(n_fine, n_coarse, P_size);
   hypre_CSRMatrixData(P) = P_data; 
   hypre_CSRMatrixI(P) = P_i; 
   hypre_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 

   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(coarse_to_fine);   
   hypre_SeqVectorDestroy(tmp_vector);   
   hypre_SeqVectorDestroy(x_vector);   
   hypre_SeqVectorDestroy(zero_vector);   

   return(0);
}            
          

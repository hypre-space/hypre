/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGBuildCoarseOperator
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMGBuildCoarseOperator(RT, A, P, RAP_ptr) 

hypre_CSRMatrix  *RT;
hypre_CSRMatrix  *A;
hypre_CSRMatrix  *P;
hypre_CSRMatrix **RAP_ptr;

{
   hypre_CSRMatrix    *RAP;
   
   double          *A_data;
   HYPRE_Int             *A_i;
   HYPRE_Int             *A_j;

   double          *P_data;
   HYPRE_Int             *P_i;
   HYPRE_Int             *P_j;

   double          *RAP_data;
   HYPRE_Int             *RAP_i;
   HYPRE_Int             *RAP_j;

   HYPRE_Int              RAP_size;
   
   hypre_CSRMatrix    *R;
   
   double          *R_data;
   HYPRE_Int             *R_i;
   HYPRE_Int             *R_j;

   HYPRE_Int             *P_marker;
   HYPRE_Int             *A_marker;

   HYPRE_Int              n_coarse;
   HYPRE_Int              n_fine;
   
   HYPRE_Int              ic, i;
   HYPRE_Int              i1, i2, i3;
   HYPRE_Int              jj1, jj2, jj3;
   
   HYPRE_Int              jj_counter;
   HYPRE_Int              jj_row_begining;
   HYPRE_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */

   double           r_entry;
   double           r_a_product;
   double           r_a_p_product;
   
   double           zero = 0.0;
   
   /*-----------------------------------------------------------------------
    *  Copy RT into R so that we have row-wise access to restriction.
    *-----------------------------------------------------------------------*/

   hypre_CSRMatrixTranspose(RT, &R, 1);   /* could call PETSc MatTranspose */

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R, A, P. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_data = hypre_CSRMatrixData(R);
   R_i    = hypre_CSRMatrixI(R);
   R_j    = hypre_CSRMatrixJ(R);

   A_data = hypre_CSRMatrixData(A);
   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   P_data = hypre_CSRMatrixData(P);
   P_i    = hypre_CSRMatrixI(P);
   P_j    = hypre_CSRMatrixJ(P);

   n_fine   = hypre_CSRMatrixNumRows(A);
   n_coarse = hypre_CSRMatrixNumRows(R);

   /*-----------------------------------------------------------------------
    *  Allocate RAP_i and marker arrays.
    *-----------------------------------------------------------------------*/

   RAP_i    = hypre_CTAlloc(HYPRE_Int, n_coarse+1);
   P_marker = hypre_CTAlloc(HYPRE_Int, n_coarse);
   A_marker = hypre_CTAlloc(HYPRE_Int, n_fine);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP and set up RAP_i
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < n_fine; i++)
   {      
      A_marker[i] = -1;
   }   

   /*-----------------------------------------------------------------------
    *  Loop over c-points.
    *-----------------------------------------------------------------------*/
    
   for (ic = 0; ic < n_coarse; ic++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, RAP_{ic,ic}.
       *--------------------------------------------------------------------*/

      P_marker[ic] = jj_counter;
      jj_row_begining = jj_counter;
      jj_counter++;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R.
       *--------------------------------------------------------------------*/
   
      for (jj1 = R_i[ic]; jj1 < R_i[ic+1]; jj1++)
      {
         i1  = R_j[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_i[i1]; jj2 < A_i[i1+1]; jj2++)
         {
            i2 = A_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P.
                *-----------------------------------------------------------*/

               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3++)
               {
                  i3 = P_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
            }
         }
      }
            
      /*--------------------------------------------------------------------
       * Set RAP_i for this row.
       *--------------------------------------------------------------------*/

      RAP_i[ic] = jj_row_begining;
      
   }
  
   RAP_i[n_coarse] = jj_counter;
 
   /*-----------------------------------------------------------------------
    *  Allocate RAP_data and RAP_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_size = jj_counter;
   RAP_data = hypre_CTAlloc(double, RAP_size);
   RAP_j    = hypre_CTAlloc(HYPRE_Int, RAP_size);

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_data and RAP_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < n_fine; i++)
   {      
      A_marker[i] = -1;
   }   
   
   /*-----------------------------------------------------------------------
    *  Loop over c-points.
    *-----------------------------------------------------------------------*/
    
   for (ic = 0; ic < n_coarse; ic++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, RAP_{ic,ic}.
       *--------------------------------------------------------------------*/

      P_marker[ic] = jj_counter;
      jj_row_begining = jj_counter;
      RAP_data[jj_counter] = zero;
      RAP_j[jj_counter] = ic;
      jj_counter++;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R.
       *--------------------------------------------------------------------*/
   
      for (jj1 = R_i[ic]; jj1 < R_i[ic+1]; jj1++)
      {
         i1  = R_j[jj1];
         r_entry = R_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_i[i1]; jj2 < A_i[i1+1]; jj2++)
         {
            i2 = A_j[jj2];
            r_a_product = r_entry * A_data[jj2];
            
            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;
               
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P.
                *-----------------------------------------------------------*/

               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3++)
               {
                  i3 = P_j[jj3];
                  r_a_p_product = r_a_product * P_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_data[jj_counter] = r_a_p_product;
                     RAP_j[jj_counter] = i3;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visted ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3++)
               {
                  i3 = P_j[jj3];
                  r_a_p_product = r_a_product * P_data[jj3];
                  RAP_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }
      }
   }

 

   RAP = hypre_CSRMatrixCreate(n_coarse, n_coarse, RAP_size);
   hypre_CSRMatrixData(RAP) = RAP_data; 
   hypre_CSRMatrixI(RAP) = RAP_i; 
   hypre_CSRMatrixJ(RAP) = RAP_j; 
   
   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R and marker arrays.
    *-----------------------------------------------------------------------*/

   hypre_CSRMatrixDestroy(R);
   hypre_TFree(P_marker);   
   hypre_TFree(A_marker);

   return(0);
   
}            




             
/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings. 
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily? 
 *--------------------------------------------------------------------------*/
          

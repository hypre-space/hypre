/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGBuildInterp
 *--------------------------------------------------------------------------*/

int
hypre_AMGBuildInterp( hypre_CSRMatrix  *A,
                   int                 *CF_marker,
                   hypre_CSRMatrix     *S,
                   hypre_CSRMatrix     **P_ptr )
{
   
   double          *A_data;
   int             *A_i;
   int             *A_j;

   double          *S_data;
   int             *S_i;
   int             *S_j;

   hypre_CSRMatrix    *P; 

   double          *P_data;
   int             *P_i;
   int             *P_j;

   int              P_size;
   
   int             *P_marker;

   int              jj_counter;
   int              jj_begin_row;
   int              jj_end_row;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine;
   int              n_coarse;

   int              strong_f_marker;

   int             *fine_to_coarse;
   int              coarse_counter;
   
   int              i,i1,i2;
   int              jj,jj1;
   
   double           diagonal;
   double           sum;
   double           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   A_data = hypre_CSRMatrixData(A);
   A_i    = hypre_CSRMatrixI(A);
   A_j    = hypre_CSRMatrixJ(A);

   S_data = hypre_CSRMatrixData(S);
   S_i    = hypre_CSRMatrixI(S);
   S_j    = hypre_CSRMatrixJ(S);

   n_fine = hypre_CSRMatrixNumRows(A);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = 0;

   fine_to_coarse = hypre_CTAlloc(int, n_fine);

   jj_counter = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i++)
   {
      
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_i[i]; jj < S_i[i+1]; jj++)
         {
            i1 = S_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_counter++;
            }
         }
      }
   }
   
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_size = jj_counter;

   P_i    = hypre_CTAlloc(int, n_fine+1);
   P_j    = hypre_CTAlloc(int, P_size);
   P_data = hypre_CTAlloc(double, P_size);

   P_marker = hypre_CTAlloc(int, n_fine);

   /*-----------------------------------------------------------------------
    *  Second Pass: Define interpolation and fill in P_data, P_i, and P_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (i = 0; i < n_fine; i++)
   {      
      P_marker[i] = -1;
   }
   
   strong_f_marker = -2;

   jj_counter = start_indexing;
   
   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i  < n_fine  ; i ++)
   {
             
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      
      if (CF_marker[i] > 0)
      {
         P_i[i] = jj_counter;
         P_j[jj_counter]    = fine_to_coarse[i];
         P_data[jj_counter] = one;
         jj_counter++;
      }
      
      
      /*--------------------------------------------------------------------
       *  If i is a f-point, build interpolation.
       *--------------------------------------------------------------------*/

      else
      {
         P_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_i[i]; jj < S_i[i+1]; jj++)
         {
            i1 = S_j[jj];   

            /*--------------------------------------------------------------
             * If nieghbor i1 is a c-point, set column number in P_j and
             * initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_j[jj_counter]    = fine_to_coarse[i1];
               P_data[jj_counter] = zero;
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If nieghbor i1 is a f-point, mark it as a strong f-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else
            {
               P_marker[i1] = strong_f_marker;
            }            
         }

         jj_end_row = jj_counter;
         
         diagonal = A_data[A_i[i]];
         
         for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
         {
            i1 = A_j[jj];

            /*--------------------------------------------------------------
             * Case 1: nieghbor i1 is a c-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_data[P_marker[i1]] += A_data[jj];
            }
 
            /*--------------------------------------------------------------
             * Case 2: nieghbor i1 is a f-point and strongly influences i,
             * distribute a_{i,i1} to c-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/

               for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++)
               {
                  i2 = A_j[jj1];
                  if (P_marker[i2] >= jj_begin_row)
                  {
                     sum += A_data[jj1];
                  }
               }
               
               distribute = A_data[jj] / sum;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++)
               {
                  i2 = A_j[jj1];
                  if (P_marker[i2] >= jj_begin_row)
                  {
                     P_data[P_marker[i2]] += distribute * A_data[jj1];
                  }
               }
            }
   
            /*--------------------------------------------------------------
             * Case 3: nieghbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else
            {
               diagonal += A_data[jj];
            }            
         }

         /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {
            P_data[jj] /= -diagonal;
         }
      }
   
      /*--------------------------------------------------------------------
       * Interpolation formula for i is done, update marker for strong
       * f connections for next i.
       *--------------------------------------------------------------------*/
   
      strong_f_marker--;
   }
  
   P_i[n_fine] = jj_counter;

   P = hypre_CreateCSRMatrix(n_fine, n_coarse, P_size);
   hypre_CSRMatrixData(P) = P_data; 
   hypre_CSRMatrixI(P) = P_i; 
   hypre_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 

   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(P_marker);   
   hypre_TFree(fine_to_coarse);   
 
   return(0);  
}            
          

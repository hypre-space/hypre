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





/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_AMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_AMGRelax( hypre_CSRMatrix *A,
                         hypre_Vector    *f,
                         HYPRE_Int            *cf_marker,
                         HYPRE_Int             relax_type,
                         HYPRE_Int             relax_points,
                         double          relax_weight,
                         hypre_Vector    *u,
                         hypre_Vector    *Vtemp )
{
   double         *A_data  = hypre_CSRMatrixData(A);
   HYPRE_Int            *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Int            *A_j     = hypre_CSRMatrixJ(A);

   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A);
   
   double         *u_data  = hypre_VectorData(u);
   double         *f_data  = hypre_VectorData(f);

   double         *Vtemp_data = hypre_VectorData(Vtemp);
   
   double          res;
	          
   HYPRE_Int             i, ii;
   HYPRE_Int             jj;
   HYPRE_Int             column;
   HYPRE_Int             relax_error = 0;

   double         *A_mat;
   double         *b_vec;

   double          zero = 0.0;
   double 	   one_minus_weight = 1.0 -relax_weight;
 
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 0 -> Jacobi
    *     relax_type = 1 -> Gauss-Seidel
    *     relax_type = 2 -> symm. Gauss-Seidel
    *     relax_type = 9 -> Direct Solve
    *-----------------------------------------------------------------------*/
   
   switch (relax_type)
   {
      case 0: /* Weighted Jacobi */
      {

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * Vtemp_data[ii];
                  }
                  u_data[i] *= one_minus_weight;
                  u_data[i] += relax_weight * res / A_data[A_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points && A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * Vtemp_data[ii];
                  }
                  u_data[i] *= one_minus_weight;
                  u_data[i] += relax_weight * res / A_data[A_i[i]];
               }
            }     
         }
         
      }
      break;
      
      case 1: /* Gauss-Seidel */
      {

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points && A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }     
         }
         
      }
      break;

      case 2: /* symm. Gauss-Seidel */
      {

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }
            for (i = n-1; i > -1; i--)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points && A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }     
            for (i = n-1; i > -1; i--)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points && A_data[A_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_i[i]+1; jj < A_i[i+1]; jj++)
                  {
                     ii = A_j[jj];
                     res -= A_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_data[A_i[i]];
               }
            }     
         }
         
      }
      break;

      case 9: /* Direct solve: use gaussian elimination */
      {

         A_mat = hypre_CTAlloc(double, n*n);
         b_vec = hypre_CTAlloc(double, n);    

         /*-----------------------------------------------------------------
          *  Load CSR matrix into A_mat.
          *-----------------------------------------------------------------*/

         for (i = 0; i < n; i++)
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               column = A_j[jj];
               A_mat[i*n+column] = A_data[jj];
            }
            b_vec[i] = f_data[i];
         }

         relax_error = gselim(A_mat,b_vec,n);

         for (i = 0; i < n; i++)
         {
            u_data[i] = b_vec[i];
         }

         hypre_TFree(A_mat); 
         hypre_TFree(b_vec);
         
      }
      break;   
   }

   return(relax_error); 
}

/*-------------------------------------------------------------------------
 *
 *                      Gaussian Elimination
 *
 *------------------------------------------------------------------------ */

HYPRE_Int gselim(A,x,n)
double *A;
double *x;
HYPRE_Int n;
{
   HYPRE_Int    err_flag = 0;
   HYPRE_Int    j,k,m;
   double factor;
   
   if (n==1)                           /* A is 1x1 */  
   {
      if (A[0] != 0.0)
      {
         x[0] = x[0]/A[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return(err_flag);
      }
   }
   else                               /* A is nxn.  Forward elimination */ 
   {
      for (k = 0; k < n-1; k++)
      {
          if (A[k*n+k] != 0.0)
          {          
             for (j = k+1; j < n; j++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k]/A[k*n+k];
                    for (m = k+1; m < n; m++)
                    {
                        A[j*n+m]  -= factor * A[k*n+m];
                    }
                                     /* Elimination step for rhs */ 
                    x[j] -= factor * x[k];              
                 }
             }
          }
       }
                                    /* Back Substitution  */
       for (k = n-1; k > 0; --k)
       {
           x[k] /= A[k*n+k];
           for (j = 0; j < k; j++)
           {
               if (A[j*n+k] != 0.0)
               {
                  x[j] -= x[k] * A[j*n+k];
               }
           }
       }
       x[0] /= A[0];
       return(err_flag);
    }
}
 

         


      

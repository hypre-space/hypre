/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_ParAMGRelax
 *--------------------------------------------------------------------------*/

int  hypre_ParAMGRelax( hypre_ParCSRMatrix *A,
                        hypre_ParVector    *f,
                        int                *cf_marker,
                        int                 relax_type,
                        int                 relax_points,
                        hypre_ParVector    *u,
                        hypre_ParVector    *Vtemp )
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double         *A_data  = hypre_CSRMatrixData(A_diag);
   int            *A_i     = hypre_CSRMatrixI(A_diag);

   int             n_global= hypre_ParCSRMatrixGlobalNumRows(A);
   int             n       = hypre_CSRMatrixNumRows(A_diag);
   int	      	   first_index = hypre_ParVectorFirstIndex(u);
   
   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   double         *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   double         *Vtemp_data = hypre_VectorData(Vtemp_local);

   hypre_CSRMatrix *A_CSR;
   int		   *A_CSR_i;   
   int		   *A_CSR_j;
   double	   *A_CSR_data;
   
   hypre_Vector    *f_vector;
   double	   *f_vector_data;

   int             i;
   int             jj;
   int             column;
   int             relax_error = 0;

   double         *A_mat;
   double         *b_vec;

   double          zero = 0.0;
   
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 0 -> Jacobi
    *     relax_type = 1 -> Gauss-Siedel <--- currently not implemented
    *     relax_type = 9 -> Direct Solve
    *-----------------------------------------------------------------------*/
   
   switch (relax_type)
   {
      case 0: /* Jacobi */
      {

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        
  	 hypre_CopyParVector(f,Vtemp); 

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
	    hypre_ParMatvec(-1.0,A, u, 1.0, Vtemp);
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_data[A_i[i]] != zero)
               {
                  u_data[i] -= Vtemp_data[i] / A_data[A_i[i]];
/*  or Jacobi relaxation
                  u_data[i] -= omega * Vtemp_data[i] / A_data[A_i[i]]; */
               }
            }
         }
      }
      break;
      
      case 9: /* Direct solve: use gaussian elimination */
      {

         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/

	 A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
	 f_vector = hypre_ParVectorToVectorAll(f);
 	 A_CSR_i = hypre_CSRMatrixI(A_CSR);
 	 A_CSR_j = hypre_CSRMatrixJ(A_CSR);
 	 A_CSR_data = hypre_CSRMatrixData(A_CSR);
   	 f_vector_data = hypre_VectorData(f_vector);

         A_mat = hypre_CTAlloc(double, n_global*n_global);
         b_vec = hypre_CTAlloc(double, n_global);    

         /*-----------------------------------------------------------------
          *  Load CSR matrix into A_mat.
          *-----------------------------------------------------------------*/

         for (i = 0; i < n_global; i++)
         {
            for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
            {
               column = A_CSR_j[jj];
               A_mat[i*n+column] = A_CSR_data[jj];
            }
            b_vec[i] = f_vector_data[i];
         }

         relax_error = gselim(A_mat,b_vec,n_global);

         for (i = 0; i < n; i++)
         {
            u_data[i] = b_vec[first_index+i];
         }

         if (n)
	 {
	    hypre_TFree(A_mat); 
            hypre_TFree(b_vec);
            hypre_DestroyCSRMatrix(A_CSR);
            hypre_DestroyVector(f_vector);
	 }
         
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

int gselim(A,x,n)
double *A;
double *x;
int n;
{
   int    err_flag = 0;
   int    j,k,m;
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
 

         


      

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
 * Relax
 *--------------------------------------------------------------------------*/

int    	 RelaxC(u,f,A,ICG,IV,
                min_point,max_point,point_type,relax_type,
                D_mat, S_vec)
Vector 	     *u;
Vector 	     *f;
Matrix       *A;
VectorInt    *ICG;
VectorInt    *IV;
int          min_point;
int          max_point;
int          point_type;
int          relax_type;
double       *D_mat;
double       *S_vec;
{
   double         *a  = MatrixData(A);
   int            *ia = MatrixIA(A);
   int            *ja = MatrixJA(A);
   int             n  = MatrixSize(A);
	          
   double         *up = VectorData(u);
   double         *fp = VectorData(f);
   int            *icg = VectorIntData(ICG);
   int            *iv = VectorIntData(IV);
   double          res;
	          
   int             i, idx, i_start, i_end;
   int             j, jj, j_low, j_high, nn;
   int             column;
   int             num_vars;
   int             relax_error = 0;

   double         *A_mat;
   double         *x_vec;
   double         *b_vec;


   /*-----------------------------------------------------------------------
    * Start Relaxation sweep.
    *-----------------------------------------------------------------------*/
   
   switch (relax_type)
   {
      case 1:                            /* Gauss-Seidel */
      {
         for (i = 0; i < n; i++)
         { 
             if ( point_type==2  ||
                 (point_type==1&&icg[i]<0)  ||
                 (point_type==3&&icg[i]>0) )
             {
                if (a[ia[i]-1] != 0.0)
                {
                   res = fp[i];
                   for (jj = ia[i]; jj < ia[i+1]-1; jj++)
	           {
	               j = ja[jj]-1;
	               res -= a[jj] * up[j];
	           }
	           up[i] = res/a[ia[i]-1];
                }
              }
         }
         return(relax_error);
      }
      break;
   case 3:                           /* Point Gauss-Seidel (simultaneously
                                        relax all variables at a grid-point.
                                        Relaxes all variables at a point if
                                        the FIRST variable at a point is
                                        of the desired point-type (i.e., 
                                        point-type = 3 and the first variable
                                        at the point is a C-variable)      */
      for (i = 0; i < max_point; ++i)
      {
          i_start = iv[i]-1;
          if ( point_type==2  ||
              (point_type==1&&icg[i_start]<=0)  ||
              (point_type==3&&icg[i_start]>0) ) 
	  {
              i_end = iv[i+1]-2;       /* iv[i+1]-1 is start of next point */
              if ((i_end == i_start) && a[ia[i_start]-1] != 0.0)
	      {
                 res = fp[i_start];
                 for (jj = ia[i_start]; jj < ia[i_start+1]-1; jj++)
	         {
	             j = ja[jj]-1;
	             res -= a[jj] * up[j];
                 }
                 up[i_start] = res/a[ia[i]-1];
              }
              else
              {
                num_vars = i_end - i_start + 1;
                for (idx = 0; idx < num_vars; ++idx)
		{
                    S_vec[idx] = 0.0;
		}
                for (idx = 0; idx < num_vars * num_vars; ++idx)
		{
                   D_mat[idx] = 0.0;
		} 
                for (idx = i_start; idx <= i_end; ++idx)
		{
                    n = idx - i_start;
                    D_mat[n*num_vars+n] = a[ia[idx]-1];
                    S_vec[n] = fp[idx];
                    j_low = ia[idx];
                    j_high = ia[idx+1]-1;
                    for (j = j_low; j < j_high; ++j)
		    { 
                        column = ja[j]-1;
                        if (column < i_start || column > i_end)
			{
                            S_vec[n] -= a[j] * up[column];
			}
                        else
			{
                            nn = column - i_start;
                            D_mat[n*num_vars+nn] = a[j];
                        }
		    }
		}
                    relax_error = gselim(D_mat,S_vec,num_vars); 
                    if (relax_error != 0) return(relax_error);
                    n = 0;
                    for (idx = i_start; idx <= i_end; ++idx)
		    {
                        up[idx] = S_vec[n++];
		    }
		
                
	      }

          }
      }
      return(relax_error);
      break;
   case 9:                           /* Direct solve: use gaussian 
                                        elimination */
      A_mat = ctalloc(double, max_point*max_point);
      b_vec = ctalloc(double, max_point);

                                    /* Load CSR matrix into A_mat */
     for (j = 0; j < max_point; ++j)
     {
       for (i = ia[j]-1; i < ia[j+1]-1; i++)
       {
           column = ja[i]-1;
           A_mat[j*max_point+column] = a[i];
       }
       b_vec[j] = fp[j];
     }
     relax_error = gselim(A_mat,b_vec,max_point);
     for (j = 0; j < max_point; j++)
     {
         up[j] = b_vec[j];
     }
     tfree(A_mat);
     return(relax_error); 
   }
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
 

         


      

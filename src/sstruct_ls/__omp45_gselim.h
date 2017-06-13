#ifndef OMP45_GSELIM_H
#define OMP45_GSELIM_H

#pragma omp declare target
static inline HYPRE_Int gselim_inline(HYPRE_Real *A,
                                      HYPRE_Real *x,
                                      HYPRE_Int n)
{
   HYPRE_Int    err_flag = 0;
   HYPRE_Int    j,k,m;
   HYPRE_Real factor;
   HYPRE_Real divA;
   
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
             divA = 1.0/A[k*n+k];
             for (j = k+1; j < n; j++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k]*divA;
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
          if (A[k*n+k] != 0.0)
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
       }
       if (A[0] != 0.0) x[0] /= A[0];
       return(err_flag);
    }
}
#pragma omp end declare target

#define gselim gselim_inline 
#endif

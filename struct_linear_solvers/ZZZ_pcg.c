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
 * Preconditioned conjugate gradient (Omin) functions
 *
 *****************************************************************************/

#include "headers.h"
#include "ZZZ_struct_pcg.h"
#include "ZZZ_pcg.h"

/*--------------------------------------------------------------------------
 * ZZZ_PCG
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>  <  (tol^2)*<C*b,b> = eps
 *
 *--------------------------------------------------------------------------*/

void
ZZZ_PCG( Vector *x, 
	 Vector *b,
	 double  tol,
	 void   *data )
{
   ZZZ_PCGData  *pcg_data      = data;

   int        max_iter     = ZZZ_PCGDataMaxIter(pcg_data);
   int        two_norm     = ZZZ_PCGDataTwoNorm(pcg_data);

   Matrix    *A            = ZZZ_PCGDataA(pcg_data);
   Vector    *p            = ZZZ_PCGDataP(pcg_data);
   Vector    *s            = ZZZ_PCGDataS(pcg_data);
   Vector    *r            = ZZZ_PCGDataR(pcg_data);

   int      (*ZZZ_PCGPrecond)()   = ZZZ_PCGDataPrecond(pcg_data);
   void      *precond_data = ZZZ_PCGDataPrecondData(pcg_data);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     bi_prod, i_prod, eps;
   
   int        i = 0;
	     
   /* logging variables */
   double    *norm_log;
   double    *rel_norm_log;
   double    *conv_rate;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   norm_log     = zzz_CTAlloc(double, max_iter);
   rel_norm_log = zzz_CTAlloc(double, max_iter);
   conv_rate    = zzz_CTAlloc(double, max_iter+1);

 
   /*-----------------------------------------------------------------------
    * Uncomment to print logging information
    *-----------------------------------------------------------------------*/

   /* printf("\nZZZ_PCG INFO:\n\n"); */


   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   if (two_norm)
   {
      /* eps = (tol^2)*<b,b> */
      bi_prod = ZZZ_InnerProd(b, b);
      eps = (tol*tol)*bi_prod;
   }
   else
   {
      /* eps = (tol^2)*<C*b,b> */
      ZZZ_InitVector(p, 0.0);
      ZZZ_PCGPrecond(p, b, 0.0, precond_data);
      bi_prod = ZZZ_InnerProd(p, b);
      eps = (tol*tol)*bi_prod;
   }

   /* r = b - Ax */
   ZZZ_CopyVector(b, r);
   ZZZ_Matvec(-1.0, A, x, 1.0, r);
 
   /* Set initial residual norm, print to log */
   norm_log[0] = sqrt(ZZZ_InnerProd(r,r));
   /* printf("\nInitial residual norm:    %e\n\n", norm_log[0]); */


   /* p = C*r */
   ZZZ_InitVector(p, 0.0);
   ZZZ_PCGPrecond(p, r, 0.0, precond_data);

   /* gamma = <r,p> */
   gamma = ZZZ_InnerProd(r,p);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      ZZZ_Matvec(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / ZZZ_InnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      ZZZ_Axpy(alpha, p, x);

      /* r = r - alpha*s */
      ZZZ_Axpy(-alpha, s, r);
	 
      /* s = C*r */
      ZZZ_InitVector(s, 0.0);
      ZZZ_PCGPrecond(s, r, 0.0, precond_data);

      /* gamma = <r,s> */
      gamma = ZZZ_InnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = ZZZ_InnerProd(r,r);
      else
	 i_prod = gamma;

#if 0
      if (two_norm)
	 printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
		i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
	 printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
		i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif
 
      /* log norm info */
      norm_log[i]     = sqrt(i_prod);
      rel_norm_log[i] = bi_prod ? sqrt(i_prod/bi_prod) : 0;

      /* check for convergence */
      if (i_prod < eps)
	 break;

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      ZZZ_ScaleVector(beta, p);   
      ZZZ_Axpy(1.0, s, p);
   }

#if 0
   if (two_norm)
      printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
   else
      printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   /*
   if (two_norm)
   {
      printf("Iters       ||r||_2    ||r||_2/||b||_2    Conv. Factor\n");
      printf("-----    ------------    ------------   ------------\n");
   }
   else
   {
      printf("Iters       ||r||_C    ||r||_C/||b||_C    Conv. Factor\n");
      printf("-----    ------------    ------------   ------------\n");
   }
   

   for (j = 1; j <= i; j++)
   {
      conv_rate[j]=norm_log[j]/norm_log[j-1];
      printf("% 5d    %e    %e    %f\n",
	      (j), norm_log[j], rel_norm_log[j], conv_rate[j]);
   }
   */
   
   /*-----------------------------------------------------------------------
    * Load logging information
    *-----------------------------------------------------------------------*/

   ZZZ_PCGDataNumIterations(pcg_data) = i;
   ZZZ_PCGDataNorm(pcg_data)          = norm_log[i];
   ZZZ_PCGDataRelNorm(pcg_data)       = rel_norm_log[i];

   zzz_TFree(norm_log);
   zzz_TFree(rel_norm_log);
   zzz_TFree(conv_rate);
}





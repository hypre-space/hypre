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
#include "pcg.h"


/*--------------------------------------------------------------------------
 * PCG
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

void     PCG(x, b, tol, data)
hypre_Vector  *x;
hypre_Vector  *b;
double   tol;
void    *data;
{
   PCGData  *pcg_data      = data;

   int        max_iter     = PCGDataMaxIter(pcg_data);
   int        two_norm     = PCGDataTwoNorm(pcg_data);

   hypre_Matrix    *A            = PCGDataA(pcg_data);
   hypre_Vector    *p            = PCGDataP(pcg_data);
   hypre_Vector    *s            = PCGDataS(pcg_data);
   hypre_Vector    *r            = PCGDataR(pcg_data);

   int      (*precond)()   = PCGDataPrecond(pcg_data);
   void      *precond_data = PCGDataPrecondData(pcg_data);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     bi_prod, i_prod, eps;
   
   int        i = 0;
	     
   /* logging variables */
   double    *norm_log;
   double    *rel_norm_log;
   double    *conv_rate;
   FILE      *log_fp;
   int        j;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   norm_log     = hypre_CTAlloc(double, max_iter);
   rel_norm_log = hypre_CTAlloc(double, max_iter);
   conv_rate    = hypre_CTAlloc(double, max_iter+1);

 
   /*-----------------------------------------------------------------------
    * Open logging file (destroy pre-existing copy)
    *-----------------------------------------------------------------------*/


   log_fp = fopen(GlobalsLogFileName, "w");
   fprintf(log_fp, "\nPCG INFO:\n\n");


   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   if (two_norm)
   {
      /* eps = (tol^2)*<b,b> */
      bi_prod = hypre_InnerProd(b, b);
      eps = (tol*tol)*bi_prod;
   }
   else
   {
      /* eps = (tol^2)*<C*b,b> */
      hypre_InitVector(p, 0.0);
      precond(p, b, 0.0, precond_data);
      bi_prod = hypre_InnerProd(p, b);
      eps = (tol*tol)*bi_prod;
   }

   /* r = b - Ax */
   hypre_CopyVector(b, r);
   hypre_Matvec(-1.0, A, x, 1.0, r);
 
   /* Set initial residual norm, print to log */
   norm_log[0] = sqrt(hypre_InnerProd(r,r));
   fprintf(log_fp, "\nInitial residual norm:    %e\n\n", norm_log[0]);


   /* p = C*r */
   hypre_InitVector(p, 0.0);
   precond(p, r, 0.0, precond_data);

   /* gamma = <r,p> */
   gamma = hypre_InnerProd(r,p);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      hypre_Matvec(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / hypre_InnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_Axpy(alpha, p, x);

      /* r = r - alpha*s */
      hypre_Axpy(-alpha, s, r);
	 
      /* s = C*r */
      hypre_InitVector(s, 0.0);
      precond(s, r, 0.0, precond_data);

      /* gamma = <r,s> */
      gamma = hypre_InnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = hypre_InnerProd(r,r);
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
      hypre_ScaleVector(beta, p);   
      hypre_Axpy(1.0, s, p);
   }

#if 1
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

/*   log_fp = fopen(GlobalsLogFileName, "a"); */

   if (two_norm)
   {
      fprintf(log_fp, "Iters       ||r||_2    ||r||_2/||b||_2    Conv. Factor\n");
      fprintf(log_fp, "-----    ------------    ------------   ------------\n");
   }
   else
   {
      fprintf(log_fp, "Iters       ||r||_C    ||r||_C/||b||_C    Conv. Factor\n");
      fprintf(log_fp, "-----    ------------    ------------   ------------\n");
   }
   

   for (j = 1; j <= i; j++)
   {
      conv_rate[j]=norm_log[j]/norm_log[j-1];
      fprintf(log_fp, "% 5d    %e    %e    %f\n",
	      (j), norm_log[j], rel_norm_log[j], conv_rate[j]);
   }
   
   fclose(log_fp);
   
   hypre_TFree(norm_log);
   hypre_TFree(rel_norm_log);
}

/*--------------------------------------------------------------------------
 * PCGSetup
 *--------------------------------------------------------------------------*/

void      PCGSetup(A, precond, precond_data, data)
hypre_Matrix   *A;
int     (*precond)();
void     *precond_data;
void     *data;
{
   PCGData  *pcg_data = data;

   double   *darray;
   int       size;


   PCGDataA(pcg_data) = A;

   size = hypre_MatrixSize(A);
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   PCGDataP(pcg_data) = hypre_NewVector(darray, size);
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   PCGDataS(pcg_data) = hypre_NewVector(darray, size);
   darray = hypre_CTAlloc(double, hypre_NDIMU(size));
   PCGDataR(pcg_data) = hypre_NewVector(darray, size);

   PCGDataPrecond(pcg_data)     = precond;
   PCGDataPrecondData(pcg_data) = precond_data;
}

/*--------------------------------------------------------------------------
 * NewPCGData
 *--------------------------------------------------------------------------*/

void     *NewPCGData(problem, solver, log_file_name)
Problem  *problem;
Solver   *solver;
char     *log_file_name;
{
   PCGData  *pcg_data;

   pcg_data = hypre_CTAlloc(PCGData, 1);

   PCGDataMaxIter(pcg_data)     = SolverPCGMaxIter(solver);
   PCGDataTwoNorm(pcg_data)     = SolverPCGTwoNorm(solver);

   PCGDataLogFileName(pcg_data) = log_file_name;

   return (void *)pcg_data;
}

/*--------------------------------------------------------------------------
 * FreePCGData
 *--------------------------------------------------------------------------*/

void   FreePCGData(data)
void  *data;
{
   PCGData  *pcg_data = data;


   if (pcg_data)
   {
      hypre_FreeVector(PCGDataP(pcg_data));
      hypre_FreeVector(PCGDataS(pcg_data));
      hypre_FreeVector(PCGDataR(pcg_data));
      hypre_TFree(pcg_data);
   }
}


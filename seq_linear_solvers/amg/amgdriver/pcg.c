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
Vector  *x;
Vector  *b;
double   tol;
void    *data;
{
   PCGData  *pcg_data      = data;

   int        max_iter     = PCGDataMaxIter(pcg_data);
   int        two_norm     = PCGDataTwoNorm(pcg_data);

   Matrix    *A            = PCGDataA(pcg_data);
   Vector    *p            = PCGDataP(pcg_data);
   Vector    *s            = PCGDataS(pcg_data);
   Vector    *r            = PCGDataR(pcg_data);

   void     (*precond)()   = PCGDataPrecond(pcg_data);
   void      *precond_data = PCGDataPrecondData(pcg_data);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     bi_prod, i_prod, eps;
   
   int        i = 0;
	     
   /* logging variables */
   double    *norm_log;
   double    *rel_norm_log;
   FILE      *log_fp;
   int        j;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   norm_log     = ctalloc(double, max_iter);
   rel_norm_log = ctalloc(double, max_iter);

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   if (two_norm)
   {
      /* eps = (tol^2)*<b,b> */
      bi_prod = InnerProd(b, b);
      eps = (tol*tol)*bi_prod;
   }
   else
   {
      /* eps = (tol^2)*<C*b,b> */
      InitVector(p, 0.0);
      precond(p, b, 0.0, precond_data);
      bi_prod = InnerProd(p, b);
      eps = (tol*tol)*bi_prod;
   }

   /* r = b - Ax */
   CopyVector(b, r);
   Matvec(-1.0, A, x, 1.0, r);

   /* p = C*r */
   InitVector(p, 0.0);
   precond(p, r, 0.0, precond_data);

   /* gamma = <r,p> */
   gamma = InnerProd(r,p);

   while (((i+1) <= max_iter) && (gamma > 0))
   {
      i++;

      /* s = A*p */
      Matvec(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / InnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      Axpy(alpha, p, x);

      /* r = r - alpha*s */
      Axpy(-alpha, s, r);
	 
      /* s = C*r */
      InitVector(s, 0.0);
      precond(s, r, 0.0, precond_data);

      /* gamma = <r,s> */
      gamma = InnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = InnerProd(r,r);
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
      norm_log[i-1]     = sqrt(i_prod);
      rel_norm_log[i-1] = bi_prod ? sqrt(i_prod/bi_prod) : 0;

      /* check for convergence */
      if (i_prod < eps)
	 break;

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      ScaleVector(beta, p);   
      Axpy(1.0, s, p);
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

   log_fp = fopen(GlobalsLogFileName, "a");

   fprintf(log_fp, "\nPCG INFO:\n\n");

   if (two_norm)
   {
      fprintf(log_fp, "Iters       ||r||_2    ||r||_2/||b||_2\n");
      fprintf(log_fp, "-----    ------------    ------------\n");
   }
   else
   {
      fprintf(log_fp, "Iters       ||r||_C    ||r||_C/||b||_C\n");
      fprintf(log_fp, "-----    ------------    ------------\n");
   }
   
   for (j = 0; j < i; j++)
   {
      fprintf(log_fp, "% 5d    %e    %e\n",
	      (j+1), norm_log[j], rel_norm_log[j]);
   }
   
   fclose(log_fp);
   
   tfree(norm_log);
   tfree(rel_norm_log);
}

/*--------------------------------------------------------------------------
 * PCGSetup
 *--------------------------------------------------------------------------*/

void      PCGSetup(A, precond, precond_data, data)
Matrix   *A;
void    (*precond)();
void     *precond_data;
void     *data;
{
   PCGData  *pcg_data = data;

   double   *darray;
   int       size;


   PCGDataA(pcg_data) = A;

   size = MatrixSize(A);
   darray = ctalloc(double, NDIMU(size));
   PCGDataP(pcg_data) = NewVector(darray, size);
   darray = ctalloc(double, NDIMU(size));
   PCGDataS(pcg_data) = NewVector(darray, size);
   darray = ctalloc(double, NDIMU(size));
   PCGDataR(pcg_data) = NewVector(darray, size);

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

   pcg_data = ctalloc(PCGData, 1);

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
      FreeVector(PCGDataP(pcg_data));
      FreeVector(PCGDataS(pcg_data));
      FreeVector(PCGDataR(pcg_data));
      tfree(pcg_data);
   }
}


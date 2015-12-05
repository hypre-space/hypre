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

   HYPRE_Int        max_iter     = PCGDataMaxIter(pcg_data);
   HYPRE_Int        two_norm     = PCGDataTwoNorm(pcg_data);

   hypre_CSRMatrix    *A         = PCGDataA(pcg_data);
   hypre_Vector    *p            = PCGDataP(pcg_data);
   hypre_Vector    *s            = PCGDataS(pcg_data);
   hypre_Vector    *r            = PCGDataR(pcg_data);

   HYPRE_Int      (*precond)()   = PCGDataPrecond(pcg_data);
   void      *precond_data = PCGDataPrecondData(pcg_data);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     bi_prod, i_prod, eps;
   
   HYPRE_Int        i = 0;
	     
   /* logging variables */
   double    *norm_log;
   double    *rel_norm_log;
   double    *conv_rate;
   FILE      *log_fp;
   HYPRE_Int        j;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   norm_log     = hypre_CTAlloc(double, max_iter);
   rel_norm_log = hypre_CTAlloc(double, max_iter);
   conv_rate    = hypre_CTAlloc(double, max_iter+1);

 
   /*-----------------------------------------------------------------------
    * Open logging file (destroy pre-existing copy)
    *-----------------------------------------------------------------------*/


   log_fp = fopen(PCGDataLogFileName(pcg_data), "w");
   hypre_fprintf(log_fp, "\nPCG INFO:\n\n");


   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   if (two_norm)
   {
      /* eps = (tol^2)*<b,b> */
      bi_prod = hypre_SeqVectorInnerProd(b, b);
      eps = (tol*tol)*bi_prod;
   }
   else
   {
      /* eps = (tol^2)*<C*b,b> */
      hypre_SeqVectorSetConstantValues(p, 0.0);
      precond(precond_data, b, p);
      bi_prod = hypre_SeqVectorInnerProd(p, b);
      eps = (tol*tol)*bi_prod;
   }

   /* r = b - Ax */
   hypre_SeqVectorCopy(b, r);
   hypre_CSRMatrixMatvec(-1.0, A, x, 1.0, r);
 
   /* Set initial residual norm, print to log */
   norm_log[0] = sqrt(hypre_SeqVectorInnerProd(r,r));
   hypre_fprintf(log_fp, "\nInitial residual norm:    %e\n\n", norm_log[0]);


   /* p = C*r */
   hypre_SeqVectorSetConstantValues(p, 0.0);
   precond(precond_data,r,p);

   /* gamma = <r,p> */
   gamma = hypre_SeqVectorInnerProd(r,p);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      hypre_CSRMatrixMatvec(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / hypre_SeqVectorInnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_SeqVectorAxpy(alpha, p, x);

      /* r = r - alpha*s */
      hypre_SeqVectorAxpy(-alpha, s, r);
	 
      /* s = C*r */
      hypre_SeqVectorSetConstantValues(s, 0.0);
      precond(precond_data,r,s);

      /* gamma = <r,s> */
      gamma = hypre_SeqVectorInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = hypre_SeqVectorInnerProd(r,r);
      else
	 i_prod = gamma;

#if 0
      if (two_norm)
	 hypre_printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
		i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
	 hypre_printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
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
      hypre_SeqVectorScale(beta, p);   
      hypre_SeqVectorAxpy(1.0, s, p);
   }

#if 1
   if (two_norm)
      hypre_printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
   else
      hypre_printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

/*   log_fp = fopen(GlobalsLogFileName, "a"); */

   if (two_norm)
   {
      hypre_fprintf(log_fp, "Iters       ||r||_2    ||r||_2/||b||_2    Conv. Factor\n");
      hypre_fprintf(log_fp, "-----    ------------    ------------   ------------\n");
   }
   else
   {
      hypre_fprintf(log_fp, "Iters       ||r||_C    ||r||_C/||b||_C    Conv. Factor\n");
      hypre_fprintf(log_fp, "-----    ------------    ------------   ------------\n");
   }
   

   for (j = 1; j <= i; j++)
   {
      conv_rate[j]=norm_log[j]/norm_log[j-1];
      hypre_fprintf(log_fp, "% 5d    %e    %e    %f\n",
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
hypre_CSRMatrix   *A;
HYPRE_Int     (*precond)();
void     *precond_data;
void     *data;
{
   PCGData  *pcg_data = data;

   HYPRE_Int       size;


   PCGDataA(pcg_data) = A;

   size = hypre_CSRMatrixNumRows(A);

   PCGDataP(pcg_data) = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(PCGDataP(pcg_data));

   PCGDataS(pcg_data) = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(PCGDataS(pcg_data));

   PCGDataR(pcg_data) = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(PCGDataR(pcg_data));


   PCGDataPrecond(pcg_data)     = precond;
   PCGDataPrecondData(pcg_data) = precond_data;
}

/*--------------------------------------------------------------------------
 * NewPCGData
 *--------------------------------------------------------------------------

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
--------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * FreePCGData
 *--------------------------------------------------------------------------

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

---------------------*/

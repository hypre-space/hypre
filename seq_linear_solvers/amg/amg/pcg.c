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

#include "amg.h"
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
Vector    *x;
Vector    *b;
double    tol;
Data     *data;
{
   PCGData  *pcg_data      = data;
#if 0
   int        max_iter     = PCGDataMaxIter(pcg_data);
   int        two_norm     = PCGDataTwoNorm(pcg_data);

   PFModule  *precond      = (instance_xtra -> precond);

   Matrix    *A            = (instance_xtra -> A);

   Vector    *r;
   Vector    *p            = (instance_xtra -> p);
   Vector    *s            = (instance_xtra -> s);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     bi_prod, i_prod, eps;
   
   int        i = 0;
	     
   double    *norm_log;
   double    *rel_norm_log;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      norm_log     = talloc(double, max_iter);
      rel_norm_log = talloc(double, max_iter);
   }

   /*-----------------------------------------------------------------------
    * Begin timing
    *-----------------------------------------------------------------------*/

   BeginTiming(public_xtra -> time_index);

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   if (zero)
      InitVector(x, 0.0);

   if (two_norm)
   {
      /* eps = (tol^2)*<b,b> */
      bi_prod = InnerProd(b, b);
      eps = (tol*tol)*bi_prod;
   }
   else
   {
      /* eps = (tol^2)*<C*b,b> */
      PFModuleInvoke(void, precond, (p, b, 0.0, 1));
      bi_prod = InnerProd(p, b);
      eps = (tol*tol)*bi_prod;
   }

   /* r = b - Ax,  (overwrite b with r) */
   Matvec(-1.0, A, x, 1.0, (r = b));

   /* p = C*r */
   PFModuleInvoke(void, precond, (p, r, 0.0, 1));

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
      PFModuleInvoke(void, precond, (s, r, 0.0, 1));

      /* gamma = <r,s> */
      gamma = InnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = InnerProd(r,r);
      else
	 i_prod = gamma;

#if 0
      if(!amps_Rank(amps_CommWorld))
      {
	 if (two_norm)
	    amps_Printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
			i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
	 else
	    amps_Printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
			i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      }
#endif
 
      /* log norm info */
      IfLogging(1)
      {
	 norm_log[i-1]     = sqrt(i_prod);
	 rel_norm_log[i-1] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }

      /* check for convergence */
      if (i_prod < eps)
	 break;

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      Scale(beta, p);   
      Axpy(1.0, s, p);
   }

#if 1
   if(!amps_Rank(amps_CommWorld))
   {
      if (two_norm)
	 amps_Printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
		     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
	 amps_Printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
		     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
   }
#endif

   /*-----------------------------------------------------------------------
    * End timing
    *-----------------------------------------------------------------------*/

   IncFLOPCount(i*2 - 1);
   EndTiming(public_xtra -> time_index);

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      FILE *log_file;
      int        j;

      log_file = OpenLogFile("PCG");

      if (two_norm)
      {
	 fprintf(log_file, "Iters       ||r||_2    ||r||_2/||b||_2\n");
	 fprintf(log_file, "-----    ------------    ------------\n");
      }
      else
      {
	 fprintf(log_file, "Iters       ||r||_C    ||r||_C/||b||_C\n");
	 fprintf(log_file, "-----    ------------    ------------\n");
      }

      for (j = 0; j < i; j++)
      {
	 fprintf(log_file, "% 5d    %e    %e\n",
		      (j+1), norm_log[j], rel_norm_log[j]);
      }

      CloseLogFile(log_file);

      tfree(norm_log);
      tfree(rel_norm_log);
   }
#endif
}

/*--------------------------------------------------------------------------
 * ReadPCGParams
 *--------------------------------------------------------------------------*/

Data  *ReadPCGParams(fp)
FILE  *fp;
{
   PCGData  *data;

   int      max_iter;
   int      two_norm;


   fscanf(fp, "%d", &max_iter);
   fscanf(fp, "%d", &two_norm);

   data = NewPCGData(max_iter, two_norm, GlobalsLogFileName);

   return data;
}

/*--------------------------------------------------------------------------
 * NewPCGData
 *--------------------------------------------------------------------------*/

Data  *NewPCGData(max_iter, two_norm, log_file_name)
int    max_iter;
int    two_norm;
char  *log_file_name;
{
   PCGData  *pcg_data;

   pcg_data = talloc(PCGData, 1);

   PCGDataMaxIter(pcg_data)     = max_iter;
   PCGDataTwoNorm(pcg_data)     = two_norm;

   PCGDataLogFileName(pcg_data) = log_file_name;

   return (Data *)pcg_data;
}

/*--------------------------------------------------------------------------
 * FreePCGData
 *--------------------------------------------------------------------------*/

void   FreePCGData(data)
Data  *data;
{
   PCGData  *pcg_data = data;


   if (pcg_data)
   {
      tfree(pcg_data);
   }
}

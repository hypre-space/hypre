/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * ConjGrad - Preconditioned conjugate gradient algorithm using the
 * ParaSails preconditioner.
 *
 *****************************************************************************/

#include "math.h"
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"

double hypre_F90_NAME_BLAS(ddot, DDOT)(HYPRE_Int *, double *, HYPRE_Int *, double *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_BLAS(dcopy, DCOPY)(HYPRE_Int *, double *, HYPRE_Int *, double *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_BLAS(dscal, DSCAL)(HYPRE_Int *, double *, double *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_BLAS(daxpy, DAXPY)(HYPRE_Int *, double *, double *, HYPRE_Int *, double *, HYPRE_Int *);

static double InnerProd(HYPRE_Int n, double *x, double *y, MPI_Comm comm)
{
    double local_result, result;

    HYPRE_Int one = 1;
    local_result = hypre_F90_NAME_BLAS(ddot, DDOT)(&n, x, &one, y, &one);

    hypre_MPI_Allreduce(&local_result, &result, 1, hypre_MPI_DOUBLE, hypre_MPI_SUM, comm);

    return result;
}

static void CopyVector(HYPRE_Int n, double *x, double *y)
{
    HYPRE_Int one = 1;
    hypre_F90_NAME_BLAS(dcopy, DCOPY)(&n, x, &one, y, &one);
}

static void ScaleVector(HYPRE_Int n, double alpha, double *x)
{
    HYPRE_Int one = 1;
    hypre_F90_NAME_BLAS(dscal, DSCAL)(&n, &alpha, x, &one);
}

static void Axpy(HYPRE_Int n, double alpha, double *x, double *y)
{
    HYPRE_Int one = 1;
    hypre_F90_NAME_BLAS(daxpy, DAXPY)(&n, &alpha, x, &one, y, &one);
}


/*--------------------------------------------------------------------------
 * PCG_ParaSails - PCG solver using ParaSails.
 * Use NULL for ps if to get unpreconditioned solve.
 * Solver will stop at step 500 if rel. resid. norm reduction is not less 
 * than 0.1 at that point.
 *--------------------------------------------------------------------------*/

void PCG_ParaSails(Matrix *mat, ParaSails *ps, double *b, double *x,
   double tol, HYPRE_Int max_iter)
{
   double *p, *s, *r;
   double alpha, beta;
   double gamma, gamma_old;
   double bi_prod, i_prod, eps;
   HYPRE_Int i = 0;
   HYPRE_Int mype;

   /* local problem size */
   HYPRE_Int n = mat->end_row - mat->beg_row + 1;

   MPI_Comm comm = mat->comm;
   hypre_MPI_Comm_rank(comm, &mype);

   /* compute square of absolute stopping threshold  */
   /* bi_prod = <b,b> */
   bi_prod = InnerProd(n, b, b, comm);
   eps = (tol*tol)*bi_prod;

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      CopyVector(n, b, x);
      return;
   }

   p = (double *) malloc(n * sizeof(double));
   s = (double *) malloc(n * sizeof(double));
   r = (double *) malloc(n * sizeof(double));

   /* r = b - Ax */
   MatrixMatvec(mat, x, r);  /* r = Ax */
   ScaleVector(n, -1.0, r);  /* r = -r */
   Axpy(n, 1.0, b, r);       /* r = r + b */
 
   /* p = C*r */
   if (ps != NULL)
      ParaSailsApply(ps, r, p);
   else
      CopyVector(n, r, p);

   /* gamma = <r,p> */
   gamma = InnerProd(n, r, p, comm);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      MatrixMatvec(mat, p, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / InnerProd(n, s, p, comm);

      gamma_old = gamma;

      /* x = x + alpha*p */
      Axpy(n, alpha, p, x);

      /* r = r - alpha*s */
      Axpy(n, -alpha, s, r);
         
      /* s = C*r */
      if (ps != NULL)
         ParaSailsApply(ps, r, s);
      else
         CopyVector(n, r, s);

      /* gamma = <r,s> */
      gamma = InnerProd(n, r, s, comm);

      /* set i_prod for convergence test */
      i_prod = InnerProd(n, r, r, comm);

#ifdef PARASAILS_CG_PRINT
      if (mype == 0 && i % 100 == 0)
         hypre_printf("Iter (%d): rel. resid. norm: %e\n", i, sqrt(i_prod/bi_prod));
#endif

      /* check for convergence */
      if (i_prod < eps)
         break;

      /* non-convergence test */
      if (i >= 1000 && i_prod/bi_prod > 0.01)
      {
         if (mype == 0)
            hypre_printf("Aborting solve due to slow or no convergence.\n");
         break;
      }
 
      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      ScaleVector(n, beta, p);   
      Axpy(n, 1.0, s, p);
   }

   free(p);
   free(s);

   /* compute exact relative residual norm */
   MatrixMatvec(mat, x, r);  /* r = Ax */
   ScaleVector(n, -1.0, r);  /* r = -r */
   Axpy(n, 1.0, b, r);       /* r = r + b */
   i_prod = InnerProd(n, r, r, comm);

   free(r);

   if (mype == 0)
      hypre_printf("Iter (%4d): computed rrn    : %e\n", i, sqrt(i_prod/bi_prod));
}

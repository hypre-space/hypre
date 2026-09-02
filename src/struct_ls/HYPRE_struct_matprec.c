/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * Matrix-preconditioner-based solver.  The error propagator is (I - wB A) where
 * B is a matrix and w is a weight.  The following routines determine B (see the
 * respective comments of each for details):
 *
 *   HYPRE_StructMatPrecSetPrec()
 *   HYPRE_StructMatPrecSetJacobi()
 *   HYPRE_StructMatPrecSetChebyshev()
 *
 * The solver data 'type' entry takes the following values:
 *
 *   0 = user provided matrix
 *   1 = weighted Jacobi (default)
 *   2 = Chebyshev
 *
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructSolver_struct
{
   /* Base solver data structure */
   hypre_Solver          base;

   hypre_StructMatrix   *A;              /* system matrix */
   hypre_StructVector   *b;              /* right-hand-side */
   hypre_StructVector   *x;              /* solution */

   HYPRE_Int             type;           /* matrix preconditioner type */
   HYPRE_Real            weight;         /* iteration weight */

   HYPRE_Real            tol;
   HYPRE_Int             max_iter;
   HYPRE_Int             zero_guess;

   /* Jacobi parameters */
   HYPRE_Int             jacobi_steps;
   HYPRE_Real            jacobi_weight;

   hypre_StructMatrix   *B;              /* precond matrix: error propagator = (I - wB A) */
   hypre_StructVector   *r;              /* residual vector */

   void                 *Ax_matvec_data;
   void                 *Br_matvec_data;

   /* log info (always logged) */
   HYPRE_Int             num_iterations;
   HYPRE_Int             print_level;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int             logging;
   HYPRE_Real           *norms;
   HYPRE_Real           *rel_norms;

} hypre_StructSolver;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecCreate( MPI_Comm            comm,
                           HYPRE_StructSolver *solver_ptr )
{
   hypre_StructSolver *solver;
   hypre_Solver       *base;

   solver = hypre_CTAlloc(hypre_StructSolver, 1, HYPRE_MEMORY_HOST);
   base        = (hypre_Solver *) solver;

   /* Set base solver function pointers */
   hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_StructMatPrecSetup;
   hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_StructMatPrecSolve;
   hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) HYPRE_StructMatPrecDestroy;

   (solver -> A)              = NULL;
   (solver -> b)              = NULL;
   (solver -> x)              = NULL;
   (solver -> type)           = 1;      /* default is Jacobi */
   (solver -> weight)         = 1.0;
   (solver -> tol)            = 1.0e-6;
   (solver -> max_iter)       = 100;
   (solver -> zero_guess)     = 0;
   (solver -> jacobi_steps)   = 1;
   (solver -> jacobi_weight)  = 1.0;
   (solver -> B)              = NULL;
   (solver -> r)              = NULL;
   (solver -> Ax_matvec_data) = NULL;
   (solver -> Br_matvec_data) = NULL;
   (solver -> num_iterations) = 0;
   (solver -> print_level)    = 0;
   (solver -> logging)        = 0;
   (solver -> norms)          = NULL;
   (solver -> rel_norms)      = NULL;

   *solver_ptr = solver;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecDestroy( HYPRE_StructSolver solver )
{
   if (solver)
   {
      if ((solver -> logging) > 0)
      {
         hypre_TFree(solver -> norms, HYPRE_MEMORY_HOST);
         hypre_TFree(solver -> rel_norms, HYPRE_MEMORY_HOST);
      }
      hypre_StructMatrixDestroy(solver -> A);
      hypre_StructVectorDestroy(solver -> b);
      hypre_StructVectorDestroy(solver -> x);
      hypre_StructMatrixDestroy(solver -> B);
      hypre_StructVectorDestroy(solver -> r);
      hypre_StructMatvecDestroy(solver -> Ax_matvec_data);
      hypre_StructMatvecDestroy(solver -> Br_matvec_data);

      hypre_TFree(solver, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSetup( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x )
{
   HYPRE_Int             type     = (solver -> type);
   HYPRE_Int             max_iter = (solver -> max_iter);
   hypre_StructMatrix   *B        = (solver -> B);
   hypre_StructVector   *r        = (solver -> r);

   /* Set A, b, x references */
   (solver -> A) = hypre_StructMatrixRef(A);
   (solver -> b) = hypre_StructVectorRef(b);
   (solver -> x) = hypre_StructVectorRef(x);

   /* Create residual vector r */
   HYPRE_StructVectorCreate(hypre_StructVectorComm(b), hypre_StructVectorGrid(b), &r);
   HYPRE_StructVectorInitialize(r);
   HYPRE_StructVectorAssemble(r);
   (solver -> r) = r;

   /* Set memory modes for vectors x and r */
   hypre_StructVectorSetMemoryMode(x, 2);
   hypre_StructVectorSetMemoryMode(r, 2);

   /*-----------------------------------------------------
    * Set up preconditioner B
    *-----------------------------------------------------*/

   switch (type)
   {
      case 0:  // User defined
      {
         break;
      }

      case 1:  // Jacobi
      {
         hypre_StructMatPrecSetupJacobi(solver);
         break;
      }

      case 2:  // Chebyshev TODO
      {
         hypre_StructMatPrecSetupChebyshev(solver);
         break;
      }

      default:  // error
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid StructMatPrec preconditioner type");
         return hypre_error_flag;
      }

   }

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((solver -> logging) > 0)
   {
      (solver -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
      (solver -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------
    * Setup matvec for A*x and A*r
    *-----------------------------------------------------*/

   if ((solver -> tol) > 0.0)
   {
      (solver -> Ax_matvec_data) = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup((solver -> Ax_matvec_data), A, x);
      (solver -> Br_matvec_data) = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup((solver -> Br_matvec_data), B, r);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSolve( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x )
{
   HYPRE_Real            tol            = (solver -> tol);
   HYPRE_Int             max_iter       = (solver -> max_iter);
   HYPRE_Int             zero_guess     = (solver -> zero_guess);
   hypre_StructMatrix   *B              = (solver -> B);
   hypre_StructVector   *r              = (solver -> r);
   void                 *Ax_matvec_data = (solver -> Ax_matvec_data);
   void                 *Br_matvec_data = (solver -> Br_matvec_data);
   HYPRE_Int             logging        = (solver -> logging);
   HYPRE_Real           *norms          = (solver -> norms);
   HYPRE_Real           *rel_norms      = (solver -> rel_norms);
   HYPRE_Real            b_dot_b = 0.0, r_dot_r = 0.0, eps = 0.0;

   HYPRE_Int             iter;

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("StructMatPrec-Solve");

   /* Reset A, b, x, num_iterations */
   hypre_StructMatrixDestroy(solver -> A);
   hypre_StructVectorDestroy(solver -> b);
   hypre_StructVectorDestroy(solver -> x);
   (solver -> A) = hypre_StructMatrixRef(A);
   (solver -> b) = hypre_StructVectorRef(b);
   (solver -> x) = hypre_StructVectorRef(x);
   (solver -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = hypre_StructInnerProd(b, b);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (!(b_dot_b > 0.0))
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
   }

   /*-----------------------------------------------------
    * Do iterations
    *-----------------------------------------------------*/

   for (iter = 0; iter < max_iter; iter++)
   {
      /* compute residual (r = b - Ax) */
      hypre_StructMatvecCompute(Ax_matvec_data, -1.0, A, x, 1.0, b, r);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = hypre_StructInnerProd(r, r);

         if (logging > 0)
         {
            norms[iter]     = hypre_sqrt(r_dot_r);
            rel_norms[iter] = hypre_sqrt(r_dot_r / b_dot_b);
         }

         if (r_dot_r / b_dot_b < eps)
         {
            break;
         }
      }

      /* compute next iterate */
      hypre_StructMatvecCompute(Br_matvec_data, 1.0, B, r, 1.0, x, x);
   }

   (solver -> num_iterations) = iter;
   hypre_StructMatPrecPrintLogging(solver);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set B to be the matrix polynomial corresponding to m+1 steps of weighted
 * Jacobi where m+1 = 'steps', w = 'weight', and
 *
 *   B = ( c0 I + c1 T + ... + cm T^m ) S
 *   S = w D^{-1}, D = diag(A)
 *   T = S A
 *   c = last m binomial coefficients of -(I - T)^(m+1)
 *
 * An alternative approach is
 *
 *   B = ( I + T + ... + T^m ) S
 *   S = w D^{-1}, D = diag(A)
 *   T = ( I - S A )
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatPrecSetupJacobi( HYPRE_StructSolver solver )
{
   hypre_StructMatrix  *A      = (solver -> A);
   HYPRE_Int            steps  = (solver -> jacobi_steps);
   HYPRE_Real           weight = (solver -> jacobi_weight);

   HYPRE_Int            m, k, coeff;
   HYPRE_Complex       *coeffs;
   hypre_StructMatrix  *B, *S, *T, *P;

   if (steps < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Must have at least one step in Jacobi MatPrec");
      return hypre_error_flag;
   }

   m = steps - 1;

   /* Get S = w diag(A)^{-1} */
   hypre_StructMatrixGetDiagMat(A, weight, 1, &S);

   if (m == 0)
   {
      (solver -> B) = S;
      return hypre_error_flag;
   }

   /* Compute T = S A */
   hypre_StructMatmat(S, A, &T);

   /* Compute polynomial coefficients from binomial coefficients */
   coeffs = hypre_TAlloc(HYPRE_Complex, m + 1, HYPRE_MEMORY_HOST);
   coeff = 1;
   for (k = 1; k < (m + 2); k++)
   {
      coeff = (coeff * (m + 2 - k)) / k;       /* kth binomial coefficient for (1+x)^(m+1) */
      coeffs[k - 1] = k % 2 ? coeff : -coeff;  /* polynomial coefficient for B */
   }

   /* Compute B */
   hypre_StructMatrixPoly(T, m, coeffs, &P);
   hypre_StructMatrixDestroy(T);
   hypre_StructMatmat(P, S, &B);
   hypre_StructMatrixDestroy(P);
   hypre_StructMatrixDestroy(S);

   (solver -> B) = B;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_StructMatPrecSetJacobi( HYPRE_StructSolver solver,
                              HYPRE_Int          steps,
                              HYPRE_Real         weight )
{
   (solver -> type)          = 1;
   (solver -> jacobi_steps)  = steps;
   (solver -> jacobi_weight) = weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set B to be the matrix polynomial for Chebyshev
 * TODO
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatPrecSetupChebyshev( HYPRE_StructSolver solver )
{
   return hypre_error_flag;
}

HYPRE_Int
HYPRE_StructMatPrecSetChebyshev( HYPRE_StructSolver solver,
                                 HYPRE_Int          steps )
{
   (solver -> type) = 2;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSetTol( HYPRE_StructSolver solver,
                           HYPRE_Real         tol )
{
   (solver -> tol) = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecGetTol( HYPRE_StructSolver solver,
                           HYPRE_Real        *tol )
{
   *tol = (solver -> tol);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSetMaxIter( HYPRE_StructSolver solver,
                               HYPRE_Int          max_iter )
{
   (solver -> max_iter) = max_iter;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecGetMaxIter( HYPRE_StructSolver solver,
                               HYPRE_Int         *max_iter )
{
   *max_iter = (solver -> max_iter);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSetZeroGuess( HYPRE_StructSolver solver )
{
   (solver -> zero_guess) = 1;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecGetZeroGuess( HYPRE_StructSolver solver,
                                 HYPRE_Int         *zero_guess )
{
   *zero_guess = (solver -> zero_guess);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecSetNonZeroGuess( HYPRE_StructSolver solver )
{
   (solver -> zero_guess) = 0;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecGetNumIterations( HYPRE_StructSolver  solver,
                                     HYPRE_Int          *num_iterations )
{
   *num_iterations = (solver -> num_iterations);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatPrecGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                 HYPRE_Real         *norm )
{
   HYPRE_Int       max_iter        = (solver -> max_iter);
   HYPRE_Int       num_iterations  = (solver -> num_iterations);
   HYPRE_Int       logging         = (solver -> logging);
   HYPRE_Real     *rel_norms       = (solver -> rel_norms);

   if (logging > 0)
   {
      if (max_iter == 0)
      {
         hypre_error_in_arg(1);
      }
      else if (num_iterations == max_iter)
      {
         *norm = rel_norms[num_iterations - 1];
      }
      else
      {
         *norm = rel_norms[num_iterations];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatPrecPrintLogging( HYPRE_StructSolver  solver )
{
   return hypre_error_flag;
}

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * a few more relaxation schemes: Chebychev, FCF-Jacobi, CG  -
 * these do not go through the CF interface (hypre_BoomerAMGRelaxIF)
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "float.h"

/******************************************************************************
 *
 * use Gershgorin discs to estimate smallest and largest eigenvalues
 * A is assumed to be symmetric
 * For SPD matrix, it returns [0, max_eig = max (aii + ri)],
 *                 ri is radius of disc centered at a_ii
 * For SND matrix, it returns [min_eig = min (aii - ri), 0]
 *
 * scale > 0: compute eigen estimate of D^{-1/2}*A*D^{-1/2}, where
 *            D = diag(A) for SPD matrix, D = -diag(A) for SND
 *
 * scale = 1: The algorithm is performed on D^{-1}*A, since it
 *            has the same eigenvalues as D^{-1/2}*A*D^{-1/2}
 * scale = 2: The algorithm is performed on D^{-1/2}*A*D^{-1/2} (TODO)
 *
 *****************************************************************************/
HYPRE_Int
hypre_ParCSRMaxEigEstimateHost( hypre_ParCSRMatrix *A,       /* matrix to relax with */
                                HYPRE_Int           scale,   /* scale by diagonal?   */
                                HYPRE_Real         *max_eig,
                                HYPRE_Real         *min_eig )
{
   HYPRE_Int   A_num_rows  = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int  *A_diag_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int  *A_diag_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int  *A_offd_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));
   HYPRE_Real *diag        = NULL;
   HYPRE_Int   i, j;
   HYPRE_Real  e_max = 0.0;
   HYPRE_Real  e_min = 0.0;
   HYPRE_Real  send_buf[2], recv_buf[2];

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   if (scale > 1)
   {
      diag = hypre_TAlloc(HYPRE_Real, A_num_rows, memory_location);
   }

   for (i = 0; i < A_num_rows; i++)
   {
      HYPRE_Real a_ii = 0.0, r_i = 0.0, lower, upper;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         if (A_diag_j[j] == i)
         {
            a_ii = A_diag_data[j];
         }
         else
         {
            r_i += hypre_abs(A_diag_data[j]);
         }
      }

      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         r_i += hypre_abs(A_offd_data[j]);
      }

      lower = a_ii - r_i;
      upper = a_ii + r_i;

      if (scale == 1)
      {
         lower /= hypre_abs(a_ii);
         upper /= hypre_abs(a_ii);
      }

      if (i)
      {
         e_max = hypre_max(e_max, upper);
         e_min = hypre_min(e_min, lower);
      }
      else
      {
         e_max = upper;
         e_min = lower;
      }
   }

   send_buf[0] = -e_min;
   send_buf[1] =  e_max;

   /* get e_min e_max across procs */
   hypre_MPI_Allreduce(send_buf, recv_buf, 2, HYPRE_MPI_REAL, hypre_MPI_MAX,
                       hypre_ParCSRMatrixComm(A));

   e_min = -recv_buf[0];
   e_max =  recv_buf[1];

   /* return */
   if ( hypre_abs(e_min) > hypre_abs(e_max) )
   {
      *min_eig = e_min;
      *max_eig = hypre_min(0.0, e_max);
   }
   else
   {
      *min_eig = hypre_max(e_min, 0.0);
      *max_eig = e_max;
   }

   hypre_TFree(diag, memory_location);

   return hypre_error_flag;
}

/**
 * @brief Estimates the max eigenvalue using infinity norm. Will determine
 * whether or not to use host or device internally
 *
 * @param[in] A Matrix to relax with
 * @param[in] to scale by diagonal
 * @param[out] Maximum eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimate(hypre_ParCSRMatrix *A, /* matrix to relax with */
                           HYPRE_Int scale, /* scale by diagonal?*/
                           HYPRE_Real *max_eig,
                           HYPRE_Real *min_eig)
{
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate");
   HYPRE_Int ierr = 0;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_ParCSRMaxEigEstimateDevice(A, scale, max_eig, min_eig);
   }
   else
#endif
   {
      ierr = hypre_ParCSRMaxEigEstimateHost(A, scale, max_eig, min_eig);
   }
   hypre_GpuProfilingPopRange();
   return ierr;
}

/**
 *  @brief Uses CG to get the eigenvalue estimate. Will determine whether to use
 *  host or device internally
 *
 *  @param[in] A Matrix to relax with
 *  @param[in] scale Gets the eigenvalue est of D^{-1/2} A D^{-1/2}
 *  @param[in] max_iter Maximum number of iterations for CG
 *  @param[out] max_eig Estimated max eigenvalue
 *  @param[out] min_eig Estimated min eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimateCG(hypre_ParCSRMatrix *A,     /* matrix to relax with */
                             HYPRE_Int           scale, /* scale by diagonal?*/
                             HYPRE_Int           max_iter,
                             HYPRE_Real         *max_eig,
                             HYPRE_Real         *min_eig)
{
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimateCG");
   HYPRE_Int             ierr = 0;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_ParCSRMaxEigEstimateCGDevice(A, scale, max_iter, max_eig, min_eig);
   }
   else
#endif
   {
      ierr = hypre_ParCSRMaxEigEstimateCGHost(A, scale, max_iter, max_eig, min_eig);
   }
   hypre_GpuProfilingPopRange();
   return ierr;
}

/**
 *  @brief Uses CG to get the eigenvalue estimate on the host
 *
 *  @param[in] A Matrix to relax with
 *  @param[in] scale Gets the eigenvalue est of D^{-1/2} A D^{-1/2}
 *  @param[in] max_iter Maximum number of iterations for CG
 *  @param[out] max_eig Estimated max eigenvalue
 *  @param[out] min_eig Estimated min eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimateCGHost( hypre_ParCSRMatrix *A,     /* matrix to relax with */
                                  HYPRE_Int           scale, /* scale by diagonal?*/
                                  HYPRE_Int           max_iter,
                                  HYPRE_Real         *max_eig,
                                  HYPRE_Real         *min_eig )
{
   HYPRE_Int i, j, err;
   hypre_ParVector *p;
   hypre_ParVector *s;
   hypre_ParVector *r;
   hypre_ParVector *ds;
   hypre_ParVector *u;

   HYPRE_Real *tridiag = NULL;
   HYPRE_Real *trioffd = NULL;

   HYPRE_Real lambda_max ;
   HYPRE_Real beta, gamma = 0.0, alpha, sdotp, gamma_old, alphainv;
   HYPRE_Real lambda_min;
   HYPRE_Real *s_data, *p_data, *ds_data, *u_data;
   HYPRE_Int local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   /* check the size of A - don't iterate more than the size */
   HYPRE_BigInt size = hypre_ParCSRMatrixGlobalNumRows(A);

   if (size < (HYPRE_BigInt) max_iter)
   {
      max_iter = (HYPRE_Int) size;
   }

   /* create some temp vectors: p, s, r , ds, u*/
   r = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(r);

   p = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(p);

   s = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(s);

   ds = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                              hypre_ParCSRMatrixGlobalNumRows(A),
                              hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(ds);

   u = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(u);

   /* point to local data */
   s_data = hypre_VectorData(hypre_ParVectorLocalVector(s));
   p_data = hypre_VectorData(hypre_ParVectorLocalVector(p));
   ds_data = hypre_VectorData(hypre_ParVectorLocalVector(ds));
   u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

   /* make room for tri-diag matrix */
   tridiag = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   trioffd = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }

   /* set residual to random */
   hypre_ParVectorSetRandomValues(r, 1);

   if (scale)
   {
      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 4);
   }
   else
   {
      /* set ds to 1 */
      hypre_ParVectorSetConstantValues(ds, 1.0);
   }

   /* gamma = <r,Cr> */
   gamma = hypre_ParVectorInnerProd(r, p);

   /* for the initial filling of the tridiag matrix */
   beta = 1.0;

   i = 0;
   while (i < max_iter)
   {
      /* s = C*r */
      /* TO DO:  C = diag scale */
      hypre_ParVectorCopy(r, s);

      /*gamma = <r,Cr> */
      gamma_old = gamma;
      gamma = hypre_ParVectorInnerProd(r, s);

      if (gamma < HYPRE_REAL_EPSILON)
      {
         break;
      }

      if (i == 0)
      {
         beta = 1.0;
         /* p_0 = C*r */
         hypre_ParVectorCopy(s, p);
      }
      else
      {
         /* beta = gamma / gamma_old */
         beta = gamma / gamma_old;

         /* p = s + beta p */
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
         for (j = 0; j < local_size; j++)
         {
            p_data[j] = s_data[j] + beta * p_data[j];
         }
      }

      if (scale)
      {
         /* s = D^{-1/2}A*D^{-1/2}*p */
         for (j = 0; j < local_size; j++)
         {
            u_data[j] = ds_data[j] * p_data[j];
         }
         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);
         for (j = 0; j < local_size; j++)
         {
            s_data[j] = ds_data[j] * s_data[j];
         }
      }
      else
      {
         /* s = A*p */
         hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, s);
      }

      /* <s,p> */
      sdotp =  hypre_ParVectorInnerProd(s, p);

      /* alpha = gamma / <s,p> */
      alpha = gamma / sdotp;

      /* get tridiagonal matrix */
      alphainv = 1.0 / alpha;

      tridiag[i + 1] = alphainv;
      tridiag[i] *= beta;
      tridiag[i] += alphainv;

      trioffd[i + 1] = alphainv;
      trioffd[i] *= hypre_sqrt(beta);

      /* x = x + alpha*p */
      /* don't need */

      /* r = r - alpha*s */
      hypre_ParVectorAxpy(-alpha, s, r);

      i++;
   }

   /* eispack routine - eigenvalues return in tridiag and ordered*/
   hypre_LINPACKcgtql1(&i, tridiag, trioffd, &err);

   lambda_max = tridiag[i - 1];
   lambda_min = tridiag[0];
   /* hypre_printf("linpack max eig est = %g\n", lambda_max);*/
   /* hypre_printf("linpack min eig est = %g\n", lambda_min);*/

   hypre_TFree(tridiag, HYPRE_MEMORY_HOST);
   hypre_TFree(trioffd, HYPRE_MEMORY_HOST);

   hypre_ParVectorDestroy(r);
   hypre_ParVectorDestroy(s);
   hypre_ParVectorDestroy(p);
   hypre_ParVectorDestroy(ds);
   hypre_ParVectorDestroy(u);

   /* return */
   *max_eig = lambda_max;
   *min_eig = lambda_min;

   return hypre_error_flag;
}

/******************************************************************************
Chebyshev relaxation

Can specify order 1-4 (this is the order of the resid polynomial)- here we
explicitly code the coefficients (instead of iteratively determining)

variant 0: standard chebyshev
this is rlx 11 if scale = 0, and 16 if scale == 1

variant 1: modified cheby: T(t)* f(t) where f(t) = (1-b/t)
this is rlx 15 if scale = 0, and 17 if scale == 1

ratio indicates the percentage of the whole spectrum to use (so .5
means half, and .1 means 10percent)
*******************************************************************************/

HYPRE_Int
hypre_ParCSRRelax_Cheby(hypre_ParCSRMatrix *A, /* matrix to relax with */
                        hypre_ParVector    *f, /* right-hand side */
                        HYPRE_Real          max_eig,
                        HYPRE_Real          min_eig,
                        HYPRE_Real          fraction,
                        HYPRE_Int           order, /* polynomial order */
                        HYPRE_Int           scale, /* scale by diagonal?*/
                        HYPRE_Int           variant,
                        hypre_ParVector    *u, /* initial/updated approximation */
                        hypre_ParVector    *v, /* temporary vector */
                        hypre_ParVector    *r /*another temp vector */)
{
   HYPRE_Real *coefs   = NULL;
   HYPRE_Real *ds_data = NULL;

   hypre_ParVector *tmp_vec    = NULL;
   hypre_ParVector *orig_u_vec = NULL;

   hypre_ParCSRRelax_Cheby_Setup(A, max_eig, min_eig, fraction, order, scale, variant, &coefs,
                                 &ds_data);

   orig_u_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(orig_u_vec, hypre_ParCSRMatrixMemoryLocation(A));

   if (scale)
   {
      tmp_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize_v2(tmp_vec, hypre_ParCSRMatrixMemoryLocation(A));
   }
   hypre_ParCSRRelax_Cheby_Solve(A, f, ds_data, coefs, order, scale, variant, u, v, r, orig_u_vec,
                                 tmp_vec);

   hypre_TFree(ds_data, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_TFree(coefs, HYPRE_MEMORY_HOST);
   hypre_ParVectorDestroy(orig_u_vec);
   hypre_ParVectorDestroy(tmp_vec);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * CG Smoother
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRRelax_CG( HYPRE_Solver        solver,
                      hypre_ParCSRMatrix *A,
                      hypre_ParVector    *f,
                      hypre_ParVector    *u,
                      HYPRE_Int           num_its)
{

   HYPRE_PCGSetMaxIter(solver, num_its); /* max iterations */
   HYPRE_PCGSetTol(solver, 0.0); /* max iterations */
   HYPRE_ParCSRPCGSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)f, (HYPRE_ParVector)u);

#if 0
   {
      HYPRE_Int myid;
      HYPRE_Int num_iterations;
      HYPRE_Real final_res_norm;

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("            -----CG PCG Iterations = %d\n", num_iterations);
         hypre_printf("            -----CG PCG Final Relative Residual Norm = %e\n", final_res_norm);
      }
   }
#endif

   return hypre_error_flag;
}


/* tql1.f --

  this is the eispack translation - from Barry Smith in Petsc

  Note that this routine always uses real numbers (not complex) even
  if the underlying matrix is Hermitian. This is because the Lanczos
  process applied to Hermitian matrices always produces a real,
  symmetric tridiagonal matrix.
*/

HYPRE_Int
hypre_LINPACKcgtql1(HYPRE_Int *n, HYPRE_Real *d, HYPRE_Real *e, HYPRE_Int *ierr)
{
   /* System generated locals */
   HYPRE_Int  i__1, i__2;
   HYPRE_Real d__1, d__2, c_b10 = 1.0;

   /* Local variables */
   HYPRE_Real c, f, g, h;
   HYPRE_Int  i, j, l, m;
   HYPRE_Real p, r, s, c2, c3 = 0.0;
   HYPRE_Int  l1, l2;
   HYPRE_Real s2 = 0.0;
   HYPRE_Int  ii;
   HYPRE_Real dl1, el1;
   HYPRE_Int  mml;
   HYPRE_Real tst1, tst2;

   /*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL1, */
   /*     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND */
   /*     WILKINSON. */
   /*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971). */

   /*     THIS SUBROUTINE FINDS THE EIGENVALUES OF A SYMMETRIC */
   /*     TRIDIAGONAL MATRIX BY THE QL METHOD. */

   /*     ON INPUT */

   /*        N IS THE ORDER OF THE MATRIX. */

   /*        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX. */

   /*        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX */
   /*          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY. */

   /*      ON OUTPUT */

   /*        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN */
   /*          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT AND */
   /*          ORDERED FOR INDICES 1,2,...IERR-1, BUT MAY NOT BE */
   /*          THE SMALLEST EIGENVALUES. */

   /*        E HAS BEEN DESTROYED. */

   /*        IERR IS SET TO */
   /*          ZERO       FOR NORMAL RETURN, */
   /*          J          IF THE J-TH EIGENVALUE HAS NOT BEEN */
   /*                     DETERMINED AFTER 30 ITERATIONS. */

   /*     CALLS CGPTHY FOR  DSQRT(A*A + B*B) . */

   /*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
   /*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
   */

   /*     THIS VERSION DATED AUGUST 1983. */

   /*     ------------------------------------------------------------------
   */
   HYPRE_Real ds;

   --e;
   --d;

   *ierr = 0;
   if (*n == 1)
   {
      goto L1001;
   }

   i__1 = *n;
   for (i = 2; i <= i__1; ++i)
   {
      e[i - 1] = e[i];
   }

   f = 0.;
   tst1 = 0.;
   e[*n] = 0.;

   i__1 = *n;
   for (l = 1; l <= i__1; ++l)
   {
      j = 0;
      h = (d__1 = d[l], hypre_abs(d__1)) + (d__2 = e[l], hypre_abs(d__2));
      if (tst1 < h)
      {
         tst1 = h;
      }
      /*     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT .......... */
      i__2 = *n;
      for (m = l; m <= i__2; ++m)
      {
         tst2 = tst1 + (d__1 = e[m], hypre_abs(d__1));
         if (tst2 == tst1)
         {
            goto L120;
         }
         /*     .......... E(N) IS ALWAYS ZERO,SO THERE IS NO EXIT */
         /*                THROUGH THE BOTTOM OF THE LOOP .......... */
      }
   L120:
      if (m == l)
      {
         goto L210;
      }
   L130:
      if (j == 30)
      {
         goto L1000;
      }
      ++j;
      /*     .......... FORM SHIFT .......... */
      l1 = l + 1;
      l2 = l1 + 1;
      g = d[l];
      p = (d[l1] - g) / (e[l] * 2.);
      r = hypre_LINPACKcgpthy(&p, &c_b10);
      ds = 1.0;
      if (p < 0.0) { ds = -1.0; }
      d[l] = e[l] / (p + ds * r);
      d[l1] = e[l] * (p + ds * r);
      dl1 = d[l1];
      h = g - d[l];
      if (l2 > *n)
      {
         goto L145;
      }

      i__2 = *n;
      for (i = l2; i <= i__2; ++i)
      {
         d[i] -= h;
      }

   L145:
      f += h;
      /*     .......... QL TRANSFORMATION .......... */
      p = d[m];
      c = 1.;
      c2 = c;
      el1 = e[l1];
      s = 0.;
      mml = m - l;
      /*     .......... FOR I=M-1 STEP -1 UNTIL L DO -- .......... */
      i__2 = mml;
      for (ii = 1; ii <= i__2; ++ii)
      {
         c3 = c2;
         c2 = c;
         s2 = s;
         i = m - ii;
         g = c * e[i];
         h = c * p;
         r = hypre_LINPACKcgpthy(&p, &e[i]);
         e[i + 1] = s * r;
         s = e[i] / r;
         c = p / r;
         p = c * d[i] - s * g;
         d[i + 1] = h + s * (c * g + s * d[i]);
      }

      p = -s * s2 * c3 * el1 * e[l] / dl1;
      e[l] = s * p;
      d[l] = c * p;
      tst2 = tst1 + (d__1 = e[l], hypre_abs(d__1));
      if (tst2 > tst1)
      {
         goto L130;
      }
   L210:
      p = d[l] + f;
      /*     .......... ORDER EIGENVALUES .......... */
      if (l == 1)
      {
         goto L250;
      }
      /*     .......... FOR I=L STEP -1 UNTIL 2 DO -- .......... */
      i__2 = l;
      for (ii = 2; ii <= i__2; ++ii)
      {
         i = l + 2 - ii;
         if (p >= d[i - 1])
         {
            goto L270;
         }
         d[i] = d[i - 1];
      }

   L250:
      i = 1;
   L270:
      d[i] = p;
   }

   goto L1001;
   /*     .......... SET ERROR -- NO CONVERGENCE TO AN */
   /*                EIGENVALUE AFTER 30 ITERATIONS .......... */
L1000:
   *ierr = l;
L1001:
   return 0;

} /* cgtql1_ */

HYPRE_Real
hypre_LINPACKcgpthy(HYPRE_Real *a, HYPRE_Real *b)
{
   /* System generated locals */
   HYPRE_Real ret_val, d__1, d__2, d__3;

   /* Local variables */
   HYPRE_Real p, r, s, t, u;

   /*     FINDS DSQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW */


   /* Computing MAX */
   d__1 = hypre_abs(*a), d__2 = hypre_abs(*b);
   p = hypre_max(d__1, d__2);
   if (!p)
   {
      goto L20;
   }
   /* Computing MIN */
   d__2 = hypre_abs(*a), d__3 = hypre_abs(*b);
   /* Computing 2nd power */
   d__1 = hypre_min(d__2, d__3) / p;
   r = d__1 * d__1;
L10:
   t = r + 4.;
   if (t == 4.)
   {
      goto L20;
   }
   s = r / t;
   u = s * 2. + 1.;
   p = u * p;
   /* Computing 2nd power */
   d__1 = s / u;
   r = d__1 * d__1 * r;
   goto L10;
L20:
   ret_val = p;

   return ret_val;
} /* cgpthy_ */

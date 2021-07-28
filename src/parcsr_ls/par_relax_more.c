/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

HYPRE_Int hypre_LINPACKcgtql1(HYPRE_Int*,HYPRE_Real *,HYPRE_Real *,HYPRE_Int *);
HYPRE_Real hypre_LINPACKcgpthy(HYPRE_Real*, HYPRE_Real*);

/**
 * @brief Estimates the max eigenvalue using infinity norm on the host
 *
 * @param[in] A Matrix to relax with
 * @param[in] to scale by diagonal
 * @param[out] Maximum eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimateHost(hypre_ParCSRMatrix *A,     /* matrix to relax with */
                               HYPRE_Int           scale, /* scale by diagonal?*/
                               HYPRE_Real         *max_eig)
{
   HYPRE_Real e_max;
   HYPRE_Real row_sum, max_norm;
   HYPRE_Real *A_diag_data;
   HYPRE_Real *A_offd_data;
   HYPRE_Real temp;
   HYPRE_Real diag_value;

   HYPRE_Int  pos_diag, neg_diag;
   HYPRE_Int  A_num_rows;
   HYPRE_Int *A_diag_i;
   HYPRE_Int *A_offd_i;
   HYPRE_Int  j;
   HYPRE_Int  i, start;

   /* estimate with the inf-norm of A - should be ok for SPD matrices */
   A_num_rows  = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   A_diag_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   A_offd_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
   A_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));

   max_norm = 0.0;

   pos_diag = neg_diag = 0;


   /* For Device Parallelization
    * Might be a good idea to do something like thrust::transform_reduce_by_key
    *    Transform each el into
    *       row
    *       max
    *       num_pos
    *       num_neg
    *    Then reduce by key over row
    *    with reductions
    *       max = max(a,b)
    *       num_pos = add(a,b)
    *       num_neg = add(a,b)
    * However, this increases the memory usage
    * */

   for ( i = 0; i < A_num_rows; i++ )
   {
      start = A_diag_i[i];
      diag_value = A_diag_data[start];
      if (diag_value > 0)
      {
         pos_diag++;
      }
      if (diag_value < 0)
      {
         neg_diag++;
         diag_value = -diag_value;
      }
      row_sum = diag_value;

      /*for (j = 0; j < row_length; j++)*/
      for (j = start+1; j < A_diag_i[i+1]; j++)
      {
         row_sum += fabs(A_diag_data[j]);
      }
      for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
      {
         row_sum += fabs(A_offd_data[j]);
      }
      if (scale)
      {
         if (diag_value != 0.0)
            row_sum = row_sum/diag_value;
      }
      if ( row_sum > max_norm ) max_norm = row_sum;
   }

   /* get max across procs */
   hypre_MPI_Allreduce(&max_norm, &temp, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, hypre_ParCSRMatrixComm(A));
   max_norm = temp;

   /* from Charles */
   if ( pos_diag == 0 && neg_diag > 0 ) max_norm = - max_norm;

   /* eig estimates */
   e_max = max_norm;

   /* return */
   *max_eig = e_max;

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
                           HYPRE_Real *max_eig)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate");
#endif
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   HYPRE_Int ierr = 0;
   if (exec == HYPRE_EXEC_HOST)
   {
      ierr = hypre_ParCSRMaxEigEstimateHost(A,scale,max_eig);
   }
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   else
   {
      ierr = hypre_ParCSRMaxEigEstimateDevice(A,scale,max_eig);
   }
#endif
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPopRange();
#endif
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimateCG");
#endif
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));
   HYPRE_Int             ierr = 0;
   if (exec == HYPRE_EXEC_HOST)
   {
      ierr = hypre_ParCSRMaxEigEstimateCGHost(A, scale, max_iter, max_eig, min_eig);
   }
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   else
   {
      ierr = hypre_ParCSRMaxEigEstimateCGDevice(A, scale, max_iter, max_eig, min_eig);
   }
#endif
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPopRange();
#endif
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
hypre_ParCSRMaxEigEstimateCGHost(hypre_ParCSRMatrix *A,     /* matrix to relax with */
                                 HYPRE_Int           scale, /* scale by diagonal?*/
                                 HYPRE_Int           max_iter,
                                 HYPRE_Real         *max_eig,
                                 HYPRE_Real         *min_eig)
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
   tridiag = hypre_CTAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
   trioffd = hypre_CTAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
   for (i=0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }

   /* set residual to random */
   hypre_ParVectorSetRandomValues(r,1);

   if (scale)
   {
      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 3);
   }
   else
   {
      /* set ds to 1 */
      hypre_ParVectorSetConstantValues(ds,1.0);
   }

   /* gamma = <r,Cr> */
   gamma = hypre_ParVectorInnerProd(r,p);

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
      gamma = hypre_ParVectorInnerProd(r,s);

      if (i==0)
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
         for (j=0; j < local_size; j++)
         {
            p_data[j] = s_data[j] + beta*p_data[j];
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
      sdotp =  hypre_ParVectorInnerProd(s,p);

      /* alpha = gamma / <s,p> */
      alpha = gamma/sdotp;

      /* get tridiagonal matrix */
      alphainv = 1.0/alpha;

      tridiag[i+1] = alphainv;
      tridiag[i] *= beta;
      tridiag[i] += alphainv;

      trioffd[i+1] = alphainv;
      trioffd[i] *= sqrt(beta);

      /* x = x + alpha*p */
      /* don't need */

      /* r = r - alpha*s */
      hypre_ParVectorAxpy( -alpha, s, r);

      i++;
   }

   /* eispack routine - eigenvalues return in tridiag and ordered*/
   hypre_LINPACKcgtql1(&i,tridiag,trioffd,&err);

   lambda_max = tridiag[i-1];
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

   hypre_ParCSRRelax_Cheby_Setup(A, max_eig, min_eig, fraction, order, scale, variant, &coefs, &ds_data);

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
   hypre_ParCSRRelax_Cheby_Solve(A, f, ds_data, coefs, order, scale, variant, u, v, r, orig_u_vec, tmp_vec);

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
      if (myid ==0)
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
hypre_LINPACKcgtql1(HYPRE_Int *n,HYPRE_Real *d,HYPRE_Real *e,HYPRE_Int *ierr)
{
   /* System generated locals */
   HYPRE_Int  i__1,i__2;
   HYPRE_Real d__1,d__2,c_b10 = 1.0;

   /* Local variables */
   HYPRE_Real c,f,g,h;
   HYPRE_Int  i,j,l,m;
   HYPRE_Real p,r,s,c2,c3 = 0.0;
   HYPRE_Int  l1,l2;
   HYPRE_Real s2 = 0.0;
   HYPRE_Int  ii;
   HYPRE_Real dl1,el1;
   HYPRE_Int  mml;
   HYPRE_Real tst1,tst2;

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
      h = (d__1 = d[l],fabs(d__1)) + (d__2 = e[l],fabs(d__2));
      if (tst1 < h)
      {
         tst1 = h;
      }
      /*     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT .......... */
      i__2 = *n;
      for (m = l; m <= i__2; ++m) {
         tst2 = tst1 + (d__1 = e[m],fabs(d__1));
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
      r = hypre_LINPACKcgpthy(&p,&c_b10);
      ds = 1.0; if (p < 0.0) ds = -1.0;
      d[l] = e[l] / (p + ds*r);
      d[l1] = e[l] * (p + ds*r);
      dl1 = d[l1];
      h = g - d[l];
      if (l2 > *n)
      {
         goto L145;
      }

      i__2 = *n;
      for (i = l2; i <= i__2; ++i) {
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
         r = hypre_LINPACKcgpthy(&p,&e[i]);
         e[i + 1] = s * r;
         s = e[i] / r;
         c = p / r;
         p = c * d[i] - s * g;
         d[i + 1] = h + s * (c * g + s * d[i]);
      }

      p = -s * s2 * c3 * el1 * e[l] / dl1;
      e[l] = s * p;
      d[l] = c * p;
      tst2 = tst1 + (d__1 = e[l],fabs(d__1));
      if (tst2 > tst1)
      {
         goto L130;
      }
L210:
      p = d[l] + f;
      /*     .......... ORDER EIGENVALUES .......... */
      if (l == 1) {
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
hypre_LINPACKcgpthy(HYPRE_Real *a,HYPRE_Real *b)
{
   /* System generated locals */
   HYPRE_Real ret_val,d__1,d__2,d__3;

   /* Local variables */
   HYPRE_Real p,r,s,t,u;

   /*     FINDS DSQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW */


   /* Computing MAX */
   d__1 = fabs(*a),d__2 = fabs(*b);
   p = hypre_max(d__1,d__2);
   if (!p)
   {
      goto L20;
   }
   /* Computing MIN */
   d__2 = fabs(*a),d__3 = fabs(*b);
   /* Computing 2nd power */
   d__1 = hypre_min(d__2,d__3) / p;
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

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCGRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCGRelaxWt( void       *amg_vdata,
                          HYPRE_Int   level,
                          HYPRE_Int   num_cg_sweeps,
                          HYPRE_Real *rlx_wt_ptr)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   MPI_Comm comm;
   HYPRE_Solver *smoother;
   /* Data Structure variables */

   /* hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data); */
   /* hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data); */
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   /* hypre_ParVector    **F_array = hypre_ParAMGDataFArray(amg_data); */
   /* hypre_ParVector    **U_array = hypre_ParAMGDataUArray(amg_data); */
   hypre_ParVector    *Utemp = NULL;
   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Qtemp = NULL;

   HYPRE_Int    *CF_marker;
   HYPRE_Real   *Ptemp_data;
   HYPRE_Real   *Ztemp_data;

   /* HYPRE_Int     **unknown_map_array;
   HYPRE_Int     **point_map_array;
   HYPRE_Int     **v_at_point_array; */


   HYPRE_Int      *grid_relax_type;

   /* Local variables  */
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       i, j, jj;
   HYPRE_Int       num_sweeps;
   HYPRE_Int       relax_type;
   HYPRE_Int       local_size;
   HYPRE_Int       old_size;
   HYPRE_Int       my_id = 0;
   HYPRE_Int       smooth_type;
   HYPRE_Int       smooth_num_levels;
   HYPRE_Int       smooth_option = 0;
   HYPRE_Int       needQ = 0;

   hypre_Vector *l1_norms = NULL;

   HYPRE_Real    alpha;
   HYPRE_Real    beta;
   HYPRE_Real    gamma = 1.0;
   HYPRE_Real    gammaold;

   HYPRE_Real   *tridiag;
   HYPRE_Real   *trioffd;
   HYPRE_Real    alphinv, row_sum = 0;
   HYPRE_Real    max_row_sum = 0;
   HYPRE_Real    rlx_wt = 0;
   HYPRE_Real    rlx_wt_old = 0;
   HYPRE_Real    lambda_max, lambda_max_old;
   /* HYPRE_Real    lambda_min, lambda_min_old; */

#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

#if !defined(HYPRE_USING_GPU)
   HYPRE_Int num_threads = hypre_NumThreads();
#endif

   /* Acquire data and allocate storage */

   tridiag  = hypre_CTAlloc(HYPRE_Real,  num_cg_sweeps + 1, HYPRE_MEMORY_HOST);
   trioffd  = hypre_CTAlloc(HYPRE_Real,  num_cg_sweeps + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cg_sweeps + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }

   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   Rtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Rtemp);

   Ptemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ptemp);

   Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ztemp);

   if (hypre_ParAMGDataL1Norms(amg_data) != NULL)
   {
      l1_norms = hypre_ParAMGDataL1Norms(amg_data)[level];
   }

#if !defined(HYPRE_USING_GPU)
   if (num_threads > 1)
#endif
   {
      needQ = 1;
   }

   grid_relax_type   = hypre_ParAMGDataGridRelaxType(amg_data);
   smooth_type       = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   CF_marker         = (hypre_ParAMGDataCFMarkerArray(amg_data)[level] != NULL) ?
                       hypre_IntArrayData(hypre_ParAMGDataCFMarkerArray(amg_data)[level]) : NULL;

   /* Initialize */

   Solve_err_flag = 0;

   comm = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (smooth_num_levels > level)
   {
      smoother = hypre_ParAMGDataSmoother(amg_data);
      smooth_option = smooth_type;
      if (smooth_type > 6 && smooth_type < 10)
      {
         Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                       hypre_ParCSRMatrixGlobalNumRows(A),
                                       hypre_ParCSRMatrixRowStarts(A));
         hypre_ParVectorInitialize(Utemp);
      }
   }

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   relax_type = grid_relax_type[1];
   num_sweeps = 1;

   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   old_size = hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp));
   hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)) =
      hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   Ptemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ptemp));
   Ztemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));
   /* if (level == 0)
      hypre_ParVectorCopy(hypre_ParAMGDataFArray(amg_data)[0],Rtemp);
   else
   {
      hypre_ParVectorCopy(F_array[level-1],Vtemp);
      alpha = -1.0;
      beta = 1.0;
      hypre_ParCSRMatrixMatvec(alpha, A_array[level-1], U_array[level-1],
                         beta, Vtemp);
      alpha = 1.0;
      beta = 0.0;

      hypre_ParCSRMatrixMatvecT(alpha,R_array[level-1],Vtemp,
                          beta,F_array[level]);
      hypre_ParVectorCopy(F_array[level],Rtemp);
   } */

   hypre_ParVectorSetRandomValues(Rtemp, 5128);

   if (needQ)
   {
      Qtemp = hypre_ParMultiVectorCreate(hypre_ParCSRMatrixComm(A),
                                         hypre_ParCSRMatrixGlobalNumRows(A),
                                         hypre_ParCSRMatrixRowStarts(A),
                                         needQ);
      hypre_ParVectorInitialize(Qtemp);
   }

   /*------------------------------------------------------------------
    * Do the relaxation num_sweeps times
    *-----------------------------------------------------------------*/

   for (jj = 0; jj < num_cg_sweeps; jj++)
   {
      hypre_ParVectorSetConstantValues(Ztemp, 0.0);

      for (j = 0; j < num_sweeps; j++)
      {
         if (smooth_num_levels > level)
         {
            if (smooth_option > 6)
            {
               hypre_ParVectorCopy(Rtemp, Vtemp);
               alpha = -1.0;
               beta = 1.0;
               hypre_ParCSRMatrixMatvec(alpha, A,
                                        Ztemp, beta, Vtemp);
               if (smooth_option == 8)
               {
                  HYPRE_ParCSRParaSailsSolve(smoother[level],
                                             (HYPRE_ParCSRMatrix) A,
                                             (HYPRE_ParVector) Vtemp,
                                             (HYPRE_ParVector) Utemp);
               }
               else if (smooth_option == 7)
               {
                  HYPRE_ParCSRPilutSolve(smoother[level],
                                         (HYPRE_ParCSRMatrix) A,
                                         (HYPRE_ParVector) Vtemp,
                                         (HYPRE_ParVector) Utemp);
                  hypre_ParVectorAxpy(1.0, Utemp, Ztemp);
               }
               else if (smooth_option == 9)
               {
                  HYPRE_EuclidSolve(smoother[level],
                                    (HYPRE_ParCSRMatrix) A,
                                    (HYPRE_ParVector) Vtemp,
                                    (HYPRE_ParVector) Utemp);
                  hypre_ParVectorAxpy(1.0, Utemp, Ztemp);
               }
            }
            else if (smooth_option == 6)
            {
               HYPRE_SchwarzSolve(smoother[level],
                                  (HYPRE_ParCSRMatrix) A,
                                  (HYPRE_ParVector) Rtemp,
                                  (HYPRE_ParVector) Ztemp);
            }
         }
         else
         {
            Solve_err_flag = hypre_BoomerAMGRelax(A,
                                                  Rtemp,
                                                  CF_marker,
                                                  relax_type,
                                                  0,
                                                  1.0,
                                                  1.0,
                                                  l1_norms ? hypre_VectorData(l1_norms) : NULL,
                                                  Ztemp,
                                                  Vtemp,
                                                  Qtemp);
         }

         if (Solve_err_flag != 0)
         {
            hypre_ParVectorDestroy(Ptemp);
            hypre_TFree(tridiag, HYPRE_MEMORY_HOST);
            hypre_TFree(trioffd, HYPRE_MEMORY_HOST);
            return (Solve_err_flag);
         }
      }

      gammaold = gamma;
      gamma = hypre_ParVectorInnerProd(Rtemp, Ztemp);
      if (jj == 0)
      {
         hypre_ParVectorCopy(Ztemp, Ptemp);
         beta = 1.0;
      }
      else
      {
         beta = gamma / gammaold;
         for (i = 0; i < local_size; i++)
         {
            Ptemp_data[i] = Ztemp_data[i] + beta * Ptemp_data[i];
         }
      }
      hypre_ParCSRMatrixMatvec(1.0, A, Ptemp, 0.0, Vtemp);
      alpha = gamma / (hypre_ParVectorInnerProd(Ptemp, Vtemp) + HYPRE_REAL_MIN);
      alphinv = 1.0 / (alpha + HYPRE_REAL_MIN);
      tridiag[jj + 1] = alphinv;
      tridiag[jj] *= beta;
      tridiag[jj] += alphinv;
      trioffd[jj] *= hypre_sqrt(beta);
      trioffd[jj + 1] = -alphinv;
      row_sum = hypre_abs(tridiag[jj]) + hypre_abs(trioffd[jj]);
      if (row_sum > max_row_sum) { max_row_sum = row_sum; }
      if (jj > 0)
      {
         row_sum = hypre_abs(tridiag[jj - 1]) + hypre_abs(trioffd[jj - 1])
                   + hypre_abs(trioffd[jj]);
         if (row_sum > max_row_sum) { max_row_sum = row_sum; }
         /* lambda_min_old = lambda_min; */
         lambda_max_old = lambda_max;
         rlx_wt_old = rlx_wt;
         hypre_Bisection(jj + 1, tridiag, trioffd, lambda_max_old,
                         max_row_sum, 1.e-3, jj + 1, &lambda_max);
         rlx_wt = 1.0 / lambda_max;
         /* hypre_Bisection(jj+1, tridiag, trioffd, 0.0, lambda_min_old,
            1.e-3, 1, &lambda_min);
         rlx_wt = 2.0/(lambda_min+lambda_max); */
         if (hypre_abs(rlx_wt - rlx_wt_old) < 1.e-3 )
         {
            /* if (my_id == 0) hypre_printf (" cg sweeps : %d\n", (jj+1)); */
            break;
         }
      }
      else
      {
         /* lambda_min = tridiag[0]; */
         lambda_max = tridiag[0];
      }

      hypre_ParVectorAxpy(-alpha, Vtemp, Rtemp);
   }
   /*if (my_id == 0)
     hypre_printf (" lambda-min: %f  lambda-max: %f\n", lambda_min, lambda_max);

   rlx_wt = hypre_abs(tridiag[0])+hypre_abs(trioffd[1]);

   for (i=1; i < num_cg_sweeps-1; i++)
   {
      row_sum = hypre_abs(tridiag[i]) + hypre_abs(trioffd[i]) + hypre_abs(trioffd[i+1]);
      if (row_sum > rlx_wt) rlx_wt = row_sum;
   }
   row_sum = hypre_abs(tridiag[num_cg_sweeps-1]) + hypre_abs(trioffd[num_cg_sweeps-1]);
   if (row_sum > rlx_wt) rlx_wt = row_sum;

   hypre_Bisection(num_cg_sweeps, tridiag, trioffd, 0.0, rlx_wt, 1.e-3, 1,
   &lambda_min);
   hypre_Bisection(num_cg_sweeps, tridiag, trioffd, 0.0, rlx_wt, 1.e-3,
   num_cg_sweeps, &lambda_max);
   */


   hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)) = old_size;

   hypre_ParVectorDestroy(Ztemp);
   hypre_ParVectorDestroy(Ptemp);
   hypre_ParVectorDestroy(Rtemp);

   if (Qtemp)
   {
      hypre_ParVectorDestroy(Qtemp);
   }

   hypre_TFree(tridiag, HYPRE_MEMORY_HOST);
   hypre_TFree(trioffd, HYPRE_MEMORY_HOST);

   if (smooth_option > 6 && smooth_option < 10)
   {
      hypre_ParVectorDestroy(Utemp);
   }

   *rlx_wt_ptr = rlx_wt;

   return (Solve_err_flag);
}

/*--------------------------------------------------------------------------
 * hypre_Bisection
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Bisection(HYPRE_Int n, HYPRE_Real *diag, HYPRE_Real *offd,
                HYPRE_Real y, HYPRE_Real z,
                HYPRE_Real tol, HYPRE_Int k, HYPRE_Real *ev_ptr)
{
   HYPRE_Real x;
   HYPRE_Real eigen_value;
   HYPRE_Int ierr = 0;
   HYPRE_Int sign_change = 0;
   HYPRE_Int i;
   HYPRE_Real p0, p1, p2;

   while (hypre_abs(y - z) > tol * (hypre_abs(y) + hypre_abs(z)))
   {
      x = (y + z) / 2;

      sign_change = 0;
      p0 = 1;
      p1 = diag[0] - x;
      if (p0 * p1 <= 0) { sign_change++; }
      for (i = 1; i < n; i++)
      {
         p2 = (diag[i] - x) * p1 - offd[i] * offd[i] * p0;
         p0 = p1;
         p1 = p2;
         if (p0 * p1 <= 0) { sign_change++; }
      }

      if (sign_change >= k)
      {
         z = x;
      }
      else
      {
         y = x;
      }
   }

   eigen_value = (y + z) / 2;
   *ev_ptr = eigen_value;

   return ierr;
}

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_block_mv.h"

HYPRE_Int gselim_piv(HYPRE_Real *A, HYPRE_Real *x, HYPRE_Int n);

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockRelaxIF

   This is the block version of the relaxation routines.

   A is now a Block matrix.

   CF_marker is size number of nodes

 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoomerAMGBlockRelaxIF( hypre_ParCSRBlockMatrix *A,
                                        hypre_ParVector    *f,
                                        HYPRE_Int          *cf_marker,
                                        HYPRE_Int           relax_type,
                                        HYPRE_Int           relax_order,
                                        HYPRE_Int           cycle_type,
                                        HYPRE_Real          relax_weight,
                                        HYPRE_Real          omega,
                                        hypre_ParVector    *u,
                                        hypre_ParVector    *Vtemp )
{
   HYPRE_Int i, Solve_err_flag = 0;
   HYPRE_Int relax_points[2];

   if (relax_order == 1 && cycle_type < 3)
      /* if do C/F and not on the cg */
   {
      if (cycle_type < 2) /* 0 = fine, 1 = down */
      {
         relax_points[0] = 1;
         relax_points[1] = -1;
      }
      else  /* 2 = up */
      {
         relax_points[0] = -1;
         relax_points[1] = 1;
      }
      for (i = 0; i < 2; i++)
      {
         Solve_err_flag = hypre_BoomerAMGBlockRelax(A,
                                                    f,
                                                    cf_marker,
                                                    relax_type,
                                                    relax_points[i],
                                                    relax_weight,
                                                    omega,
                                                    u,
                                                    Vtemp);
      }
   }
   else /* either on the cg or doing normal relaxation (no C/F relaxation) */
   {
      Solve_err_flag = hypre_BoomerAMGBlockRelax(A,
                                                 f,
                                                 cf_marker,
                                                 relax_type,
                                                 0,
                                                 relax_weight,
                                                 omega,
                                                 u,
                                                 Vtemp);
   }

   return Solve_err_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockRelax

 This is the block version of the relaxation routines.

 A is now a Block matrix.

 CF_marker is size number of nodes.

 *--------------------------------------------------------------------------*/
HYPRE_Int  hypre_BoomerAMGBlockRelax( hypre_ParCSRBlockMatrix *A,
                                      hypre_ParVector    *f,
                                      HYPRE_Int          *cf_marker,
                                      HYPRE_Int           relax_type,
                                      HYPRE_Int           relax_points,
                                      HYPRE_Real          relax_weight,
                                      HYPRE_Real          omega,
                                      hypre_ParVector    *u,
                                      hypre_ParVector    *Vtemp )

{
   MPI_Comm              comm = hypre_ParCSRBlockMatrixComm(A);

   hypre_CSRBlockMatrix *A_diag       = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real           *A_diag_data  = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRBlockMatrixJ(A_diag);

   hypre_CSRBlockMatrix *A_offd       = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Int            *A_offd_i     = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Real           *A_offd_data  = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int            *A_offd_j     = hypre_CSRBlockMatrixJ(A_offd);

   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle = NULL;

   HYPRE_Int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int             bnnz = block_size * block_size;

   HYPRE_BigInt          n_global;
   HYPRE_Int             n             = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_Int             num_cols_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_BigInt          first_index = hypre_ParVectorFirstIndex(u);

   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real     *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real     *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real     *Vtemp_data  = hypre_VectorData(Vtemp_local);
   HYPRE_Real     *Vext_data = NULL;
   HYPRE_Real     *v_buf_data = NULL;

   HYPRE_Real     *tmp_data;

   HYPRE_Int       size, rest, ne, ns;


   HYPRE_Int       i, j, k;
   HYPRE_Int       ii, jj;

   HYPRE_Int       relax_error = 0;
   HYPRE_Int       num_sends;
   HYPRE_Int       index, start;
   HYPRE_Int       num_procs, num_threads, my_id;

   HYPRE_Real      *res_vec, *out_vec, *tmp_vec;
   HYPRE_Real      *res0_vec, *res2_vec;
   HYPRE_Real      one_minus_weight;
   HYPRE_Real      one_minus_omega;
   HYPRE_Real      prod;

   hypre_CSRMatrix *A_CSR;
   HYPRE_Int       *A_CSR_i;
   HYPRE_Int       *A_CSR_j;
   HYPRE_Real      *A_CSR_data;

   hypre_Vector    *f_vector;
   HYPRE_Real      *f_vector_data;

   hypre_ParCSRMatrix *A_ParCSR;

   HYPRE_Real     *A_mat;
   HYPRE_Real     *b_vec;

   HYPRE_Int       column;

   /* initialize some stuff */
   one_minus_weight = 1.0 - relax_weight;
   one_minus_omega = 1.0 - omega;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   /* num_threads = hypre_NumThreads(); */
   num_threads = 1;

   res_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);
   out_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);
   tmp_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 20 -> Jacobi or CF-Jacobi

    *     relax_type = 23 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *                       with outer relaxation parameters (forward solve)
    *
    *     relax_type = 26 ->  hybrid: Jacobi off-processor,
    *                          Symm. Gauss-Seidel/ SSOR on-processor
    *                         with outer relaxation paramete
    *     relax_type = 29 -> Direct Solve
    *-----------------------------------------------------------------------*/
   switch (relax_type)
   {

      /*---------------------------------------------------------------------------
        Jacobi
        ---------------------------------------------------------------------------*/
      case 20:
      {

         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends) * block_size, HYPRE_MEMORY_HOST);
            Vext_data = hypre_CTAlloc(HYPRE_Real,  num_cols_offd * block_size, HYPRE_MEMORY_HOST);
            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
               A_offd_data = hypre_CSRBlockMatrixData(A_offd);
            }
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     v_buf_data[index++]
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                  }
               }
            }

            /* we need to use the block comm handle here - since comm_pkg is nodal based */
            comm_handle = hypre_ParCSRBlockCommHandleCreate(1, block_size, comm_pkg,
                                                            v_buf_data, Vext_data );

         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/

         for (i = 0; i < n * block_size; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         if (num_procs > 1)
         {
            hypre_ParCSRBlockCommHandleDestroy(comm_handle); /* now Vext_data is populated */
            comm_handle = NULL;
         }
         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               for (k = 0; k < block_size; k++)
               {
                  res_vec[k] = f_data[i * block_size + k];
               }
               for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
               {
                  ii = A_diag_j[jj];
                  /* res -= A_diag_data[jj] * Vtemp_data[ii]; */
                  hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                  &Vtemp_data[ii * block_size],
                                                  1.0, res_vec, block_size);
               }
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  ii = A_offd_j[jj];
                  /* res -= A_offd_data[jj] * Vext_data[ii]; */
                  hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                  &Vext_data[ii * block_size],
                                                  1.0, res_vec, block_size);
               }

               /* if diag is singular, then skip this point */
               if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                       out_vec, block_size) == 0)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     u_data[i * block_size + k] *= one_minus_weight;
                     u_data[i * block_size + k] += relax_weight * out_vec[k];
                  }

               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points)
               {

                  for (k = 0; k < block_size; k++)
                  {
                     res_vec[k] = f_data[i * block_size + k];
                  }
                  for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     /* res -= A_diag_data[jj] * Vtemp_data[ii]; */
                     hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                     &Vtemp_data[ii * block_size],
                                                     1.0, res_vec, block_size);
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     /* res -= A_offd_data[jj] * Vext_data[ii]; */
                     hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                     &Vext_data[ii * block_size],
                                                     1.0, res_vec, block_size);
                  }

                  /* if diag is singular, then skip this point */
                  if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                          out_vec, block_size) == 0)
                  {
                     for (k = 0; k < block_size; k++)
                     {
                        u_data[i * block_size + k] *= one_minus_weight;
                        u_data[i * block_size + k] += relax_weight * out_vec[k];
                     }

                  }
               }
            }
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }

         break;

         } /* end case 20 */

      /*---------------------------------------------------------------------------
        Hybrid: G-S on proc. and Jacobi off proc.
        ---------------------------------------------------------------------------*/
      case 23:
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends) * block_size, HYPRE_MEMORY_HOST);
            Vext_data = hypre_CTAlloc(HYPRE_Real,  num_cols_offd * block_size, HYPRE_MEMORY_HOST);
            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
               A_offd_data = hypre_CSRBlockMatrixData(A_offd);
            }
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     v_buf_data[index++]
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                  }
               }
            }

            /* we need to use the block comm handle here - since comm_pkg is nodal based */
            comm_handle = hypre_ParCSRBlockCommHandleCreate(1, block_size, comm_pkg,
                                                            v_buf_data, Vext_data );

         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/


         for (i = 0; i < n * block_size; i++)
         {
            Vtemp_data[i] = u_data[i];
         }

         if (num_procs > 1)
         {
            hypre_ParCSRBlockCommHandleDestroy(comm_handle); /* now Vext_data is populated */
            comm_handle = NULL;
         }


         /*-----------------------------------------------------------------
          * relax weight and omega = 1
          *-----------------------------------------------------------------*/

         if (relax_weight == 1 && omega == 1)
         {

            /*-----------------------------------------------------------------
             * Relax all points.
             *-----------------------------------------------------------------*/
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }

                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++)  /* interior points first */
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /*  res -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);

                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     } /* for loop over points */
                  } /* foor loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++)       /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                     }
                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] = out_vec[k];
                        }
                     }
                  } /* for loop over points */
               } /* end of num_threads = 1 */
            } /* end of non-CF relaxation */

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }

                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {
                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        if (cf_marker[i] == relax_points )
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);

                           }
                           /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                           /* if diag is singular, then skip this point */
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] = out_vec[k];
                              }
                           }
                        }
                     } /* loop over points */
                  } /* loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {
                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is
                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     if (cf_marker[i] == relax_points )
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);

                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     }
                  } /* end of loop over points */
               } /* end of num_threads = 1 */
            } /* end of C/F option */
         }
         else
         {
            /*-----------------------------------------------------------------
             * relax weight and omega do not = 1
             *-----------------------------------------------------------------*/
            prod = (1.0 - relax_weight * omega);
            res0_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);
            res2_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);

            if (relax_points == 0)
            {
               /*-----------------------------------------------------------------
                * Relax all points.
                *-----------------------------------------------------------------*/

               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++)  /* interior points first */
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /* res0 -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res0_vec, block_size);
                              /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                              &Vtemp_data[ii * block_size],
                                                              1.0, res2_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }

                     } /* end of loop over points */
                  } /* end of loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++)       /* interior points first */
                  {
                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                     }
                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res0 -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res0_vec, block_size);
                        /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                        &Vtemp_data[ii * block_size],
                                                        1.0, res2_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                     for (k = 0; k < block_size; k++)
                     {
                        tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                     }
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] *= prod;
                           u_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                     }
                  } /* end of loop over points */
               } /* end num_threads = 1 */
            }
            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points)
                        {
                           /*-----------------------------------------------------------
                            * If diagonal is nonzero, relax point i; otherwise, skip it.
                            *-----------------------------------------------------------*/
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                              res0_vec[k] = 0.0;
                              res2_vec[k] = 0.0;
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res0 -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res0_vec, block_size);
                                 /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                                 hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                                 &Vtemp_data[ii * block_size],
                                                                 1.0, res2_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                           /* u_data[i] *= prod;
                              u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                           for (k = 0; k < block_size; k++)
                           {
                              tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                           }
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] *= prod;
                                 u_data[i * block_size + k] += relax_weight * out_vec[k];
                              }
                           }
                        } /* end of if cf_marker */
                     } /* end loop over points */
                  } /* end loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is
                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points)
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res0 -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res0_vec, block_size);
                           /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                           &Vtemp_data[ii * block_size],
                                                           1.0, res2_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }
                     } /* end cf_marker */
                  }  /* end loop over points */
               } /* end num_threads = 1 */
            } /* end C/F option */
            hypre_TFree(res0_vec, HYPRE_MEMORY_HOST);
            hypre_TFree(res2_vec, HYPRE_MEMORY_HOST);
         } /* end of check relax weight and omega */

         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }


         break;
      }



      /*-----------------------------------------------------------------
        Hybrid: Jacobi off-processor,
        Symm. Gauss-Seidel/ SSOR on-processor
        with outer relaxation parameter
        *-----------------------------------------------------------------*/

      case 26:
      {

         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends) * block_size, HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd * block_size, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
               A_offd_data = hypre_CSRBlockMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     v_buf_data[index++]
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j) * block_size + k];
                  }
               }

            }

            /* we need to use the block comm handle here - since comm_pkg is nodal based */
            comm_handle = hypre_ParCSRBlockCommHandleCreate( 1, block_size, comm_pkg,
                                                             v_buf_data, Vext_data);


            hypre_ParCSRBlockCommHandleDestroy(comm_handle);
            comm_handle = NULL;

         }

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_weight == 1 && omega == 1)
         {
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++)   /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }

                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /* res -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }

                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];

                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     } /* end of interior points loop */

                     for (i = ne - 1; i > ns - 1; i--)   /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }

                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {

                              /* res -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res_vec, block_size);

                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);

                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     } /* end of loop over points */
                  } /* end of loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_thread ==1 */
               {
                  for (i = 0; i < n; i++)        /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                     }
                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];

                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] = out_vec[k];
                        }
                     }
                  } /* end of loop over points */
                  for (i = n - 1; i > -1; i--)   /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                     }

                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] = out_vec[k];
                        }
                     }

                  } /* end loop over points */
               }  /* end of num_threads = 1 */
            } /* end of non-CF relaxation*/

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);

                           }
                           /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                           /* if diag is singular, then skip this point */
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] = out_vec[k];
                              }
                           }
                        }
                     }

                     for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);

                           }
                           /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                           /* if diag is singular, then skip this point */
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] = out_vec[k];
                              }
                           }
                        }
                     } /* loop over pts */
                  }     /* over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);

                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     }
                  }
                  for (i = n - 1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] = out_vec[k];
                           }
                        }
                     }
                  }/* end of loop over points */
               } /* end of num_threads = 1 */
            }  /* end of C/F option */
         }
         else
         {
            /*-----------------------------------------------------------------
             * relax weight and omega do not = 1
             *-----------------------------------------------------------------*/
            prod = (1.0 - relax_weight * omega);
            res0_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);
            res2_vec = hypre_CTAlloc(HYPRE_Real,  block_size, HYPRE_MEMORY_HOST);
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0 - relax_weight * omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++)   /* interior points first */
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }

                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];

                           if (ii >= ns && ii < ne)
                           {
                              /* res0 -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res0_vec, block_size);
                              /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                              &Vtemp_data[ii * block_size],
                                                              1.0, res2_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                        }

                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);


                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }
                     }

                     for (i = ne - 1; i > ns - 1; i--)   /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }

                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /* res0 -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &u_data[ii * block_size],
                                                              1.0, res0_vec, block_size);
                              /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                              &Vtemp_data[ii * block_size],
                                                              1.0, res2_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                              &tmp_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }
                     } /* end of loop over points */
                  } /* loop over threads end */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num threads = 1 */
               {
                  for (i = 0; i < n; i++)        /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                     }
                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res0 -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res0_vec, block_size);
                        /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                        &Vtemp_data[ii * block_size],
                                                        1.0, res2_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                     for (k = 0; k < block_size; k++)
                     {
                        tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                     }
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] *= prod;
                           u_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                     }
                  }
                  for (i = n - 1; i > -1; i--)   /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k = 0; k < block_size; k++)
                     {
                        res_vec[k] = f_data[i * block_size + k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                     }
                     for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res0 -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                        &u_data[ii * block_size],
                                                        1.0, res0_vec, block_size);
                        /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                        &Vtemp_data[ii * block_size],
                                                        1.0, res2_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                        &Vext_data[ii * block_size],
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                     for (k = 0; k < block_size; k++)
                     {
                        tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                     }
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                             out_vec, block_size) == 0)
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           u_data[i * block_size + k] *= prod;
                           u_data[i * block_size + k] += relax_weight * out_vec[k];
                        }
                     }
                  }/* end of loop over points */
               }/* end num_threads = 1 */
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n / num_threads;
                     rest = n - size * num_threads;
                     if (j < rest)
                     {
                        ns = j * size + j;
                        ne = (j + 1) * size + j + 1;
                     }
                     else
                     {
                        ns = j * size + rest;
                        ne = (j + 1) * size + rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points )
                        {
                           /*-----------------------------------------------------------
                            * If diagonal is nonzero, relax point i; otherwise, skip it.
                            *-----------------------------------------------------------*/
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                              res0_vec[k] = 0.0;
                              res2_vec[k] = 0.0;
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res0 -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res0_vec, block_size);
                                 /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                                 hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                                 &Vtemp_data[ii * block_size],
                                                                 1.0, res2_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                           /* u_data[i] *= prod;
                              u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                           for (k = 0; k < block_size; k++)
                           {
                              tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                           }
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] *= prod;
                                 u_data[i * block_size + k] += relax_weight * out_vec[k];
                              }
                           }
                        }
                     }
                     for (i = ne - 1; i > ns - 1; i--) /* relax interior points */
                     {
                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points)
                        {
                           /*-----------------------------------------------------------
                            * If diagonal is nonzero, relax point i; otherwise, skip it.
                            *-----------------------------------------------------------*/
                           for (k = 0; k < block_size; k++)
                           {
                              res_vec[k] = f_data[i * block_size + k];
                              res0_vec[k] = 0.0;
                              res2_vec[k] = 0.0;
                           }
                           for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res0 -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &u_data[ii * block_size],
                                                                 1.0, res0_vec, block_size);
                                 /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                                 hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                                 &Vtemp_data[ii * block_size],
                                                                 1.0, res2_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                                 &tmp_data[ii * block_size],
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                              &Vext_data[ii * block_size],
                                                              1.0, res_vec, block_size);
                           }
                           /* u_data[i] *= prod;
                              u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                           for (k = 0; k < block_size; k++)
                           {
                              tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                           }
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                   out_vec, block_size) == 0)
                           {
                              for (k = 0; k < block_size; k++)
                              {
                                 u_data[i * block_size + k] *= prod;
                                 u_data[i * block_size + k] += relax_weight * out_vec[k];
                              }
                           }
                        }
                     }  /* end loop over points */
                  }    /* end loop over threads */
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is
                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points )
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res0 -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res0_vec, block_size);
                           /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                           &Vtemp_data[ii * block_size],
                                                           1.0, res2_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }
                     }
                  }
                  for (i = n - 1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points )
                     {
                        for (k = 0; k < block_size; k++)
                        {
                           res_vec[k] = f_data[i * block_size + k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }
                        for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res0 -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj * bnnz],
                                                           &u_data[ii * block_size],
                                                           1.0, res0_vec, block_size);
                           /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj * bnnz],
                                                           &Vtemp_data[ii * block_size],
                                                           1.0, res2_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj * bnnz],
                                                           &Vext_data[ii * block_size],
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k = 0; k < block_size; k++)
                        {
                           tmp_vec[k] =  omega * res_vec[k] + res0_vec[k] + one_minus_omega * res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec,
                                                                out_vec, block_size) == 0)
                        {
                           for (k = 0; k < block_size; k++)
                           {
                              u_data[i * block_size + k] *= prod;
                              u_data[i * block_size + k] += relax_weight * out_vec[k];
                           }
                        }
                     }
                  }  /* loop over points */
               } /* num threads = 1 */
            } /* CF option */
            hypre_TFree(res0_vec, HYPRE_MEMORY_HOST);
            hypre_TFree(res2_vec, HYPRE_MEMORY_HOST);
         } /* end of check relax weight and omega */
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }
         break;
      }

      /*---------------------------------------------------------------------------
       * Direct solve: use gaussian elimination
       *---------------------------------------------------------------------------*/
      case 29:
      {

         /* for now, we convert to a parcsr and
            then proceed as in non-block case  - shouldn't be too expensive
            since this is a small matrix.  Would be better to just store the CSR matrix
            in the amg_data structure - esp. in the case of no global partition */

         A_ParCSR =  hypre_ParCSRBlockMatrixConvertToParCSRMatrix(A);
         n_global = hypre_ParCSRMatrixGlobalNumRows(A_ParCSR);
         HYPRE_Int n_small = (HYPRE_Int) n_global; /* we expect n_global to be small at this point */

         /*  Generate CSR matrix from ParCSRMatrix A */

         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A_ParCSR);
         f_vector = hypre_ParVectorToVectorAll(f);

         if (n)
         {
            A_CSR_i = hypre_CSRMatrixI(A_CSR);
            A_CSR_j = hypre_CSRMatrixJ(A_CSR);
            A_CSR_data = hypre_CSRMatrixData(A_CSR);
            f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real,  n_small * n_small, HYPRE_MEMORY_HOST);
            b_vec = hypre_CTAlloc(HYPRE_Real,  n_small, HYPRE_MEMORY_HOST);

            /*  Load CSR matrix into A_mat. */


            for (i = 0; i < n_small; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i * n_small + column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            relax_error = gselim_piv(A_mat, b_vec, n_small);

            /* should check the relax error */

            for (i = 0; i < n; i++)
            {
               for (k = 0; k < block_size; k++)
               {
                  u_data[i * block_size + k] = b_vec[first_index + i * block_size + k];
               }
            }

            hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
            hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;

         }
         else
         {
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }

         hypre_ParCSRMatrixDestroy(A_ParCSR);
         A_ParCSR = NULL;

         break;
      }

   }


   hypre_TFree(res_vec, HYPRE_MEMORY_HOST);
   hypre_TFree(out_vec, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_vec, HYPRE_MEMORY_HOST);

   return (relax_error);

}

/*-------------------------------------------------------------------------
 *
 *                      Gaussian Elimination - with pivoting
 *
 *------------------------------------------------------------------------ */

HYPRE_Int gselim_piv(HYPRE_Real *A, HYPRE_Real *x, HYPRE_Int n)
{
   HYPRE_Int    err_flag = 0;
   HYPRE_Int    j, k, m, piv_row;
   HYPRE_Real   factor, piv, tmp;
   HYPRE_Real   eps = 1e-8;

   if (n == 1)                         /* A is 1x1 */
   {
      if (hypre_abs(A[0]) >  1e-10)
      {
         x[0] = x[0] / A[0];
         return (err_flag);
      }
      else
      {
         err_flag = 1;
         return (err_flag);
      }
   }
   else                               /* A is nxn.  Forward elimination */
   {
      for (k = 0; k < n - 1; k++)
      {
         /* we do partial pivoting for size */

         piv = A[k * n + k];
         piv_row = k;
         /* find the largest pivot in position k*/
         for (j = k + 1; j < n; j++)
         {
            if (hypre_abs(A[j * n + k]) > hypre_abs(piv))
            {
               piv =  A[j * n + k];
               piv_row = j;
            }
         }
         if (piv_row != k) /* do a row exchange  - rows k and piv_row*/
         {
            for (j = 0; j < n; j++)
            {
               tmp = A[k * n + j];
               A[k * n + j] = A[piv_row * n + j];
               A[piv_row * n + j] = tmp;
            }
            tmp = x[k];
            x[k] = x[piv_row];
            x[piv_row] = tmp;
         }


         if (hypre_abs(piv) > eps)
         {
            for (j = k + 1; j < n; j++)
            {
               if (A[j * n + k] != 0.0)
               {
                  factor = A[j * n + k] / A[k * n + k];
                  for (m = k + 1; m < n; m++)
                  {
                     A[j * n + m]  -= factor * A[k * n + m];
                  }
                  /* Elimination step for rhs */
                  x[j] -= factor * x[k];
               }
            }
         }
         else
         {
            /* hypre_printf("Matrix is nearly singular: zero pivot error\n"); */
            return (-1);
         }
      }
      /* we also need to check the pivot in the last row to see if it is zero */
      k = n - 1; /* last row */
      if ( hypre_abs(A[k * n + k]) < eps)
      {
         /* hypre_printf("Block of matrix is nearly singular: zero pivot error\n"); */
         return (-1);
      }

      /* Back Substitution  */
      for (k = n - 1; k > 0; --k)
      {
         x[k] /= A[k * n + k];
         for (j = 0; j < k; j++)
         {
            if (A[j * n + k] != 0.0)
            {
               x[j] -= x[k] * A[j * n + k];
            }
         }
      }
      x[0] /= A[0];
      return (err_flag);
   }
}

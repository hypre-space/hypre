/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "Common.h"
#include "_hypre_lapack.h"
#include "../sstruct_ls/gselim.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoomerAMGRelax( hypre_ParCSRMatrix *A,
                                 hypre_ParVector    *f,
                                 HYPRE_Int          *cf_marker,
                                 HYPRE_Int           relax_type,
                                 HYPRE_Int           relax_points,
                                 HYPRE_Real          relax_weight,
                                 HYPRE_Real          omega,
                                 HYPRE_Real         *l1_norms,
                                 hypre_ParVector    *u,
                                 hypre_ParVector    *Vtemp,
                                 hypre_ParVector    *Ztemp )
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Real      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_BigInt     global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int        n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt     first_ind = hypre_ParVectorFirstIndex(u);

   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real     *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real     *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local;
   HYPRE_Real     *Vtemp_data;
   if (relax_type != 10)
   {
      Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
      Vtemp_data = hypre_VectorData(Vtemp_local);
   }
   HYPRE_Real     *Vext_data = NULL;
   HYPRE_Real     *v_buf_data = NULL;
   HYPRE_Real     *tmp_data;

   hypre_Vector   *Ztemp_local;
   HYPRE_Real     *Ztemp_data;

   hypre_CSRMatrix *A_CSR;
   HYPRE_Int       *A_CSR_i;
   HYPRE_Int       *A_CSR_j;
   HYPRE_Real      *A_CSR_data;

   hypre_Vector    *f_vector;
   HYPRE_Real      *f_vector_data;

   HYPRE_Int        i, j, jr;
   HYPRE_Int        ii, jj;
   HYPRE_Int        ns, ne, size, rest;
   HYPRE_Int        column;
   HYPRE_Int        relax_error = 0;
   HYPRE_Int        num_sends;
   HYPRE_Int        num_recvs;
   HYPRE_Int        index, start;
   HYPRE_Int        num_procs, num_threads, my_id, ip, p;
   HYPRE_Int        vec_start, vec_len;
   hypre_MPI_Status     *status;
   hypre_MPI_Request    *requests;

   HYPRE_Real     *A_mat;
   HYPRE_Real     *b_vec;

   HYPRE_Real      zero = 0.0;
   HYPRE_Real      res, res0, res2;
   HYPRE_Real      one_minus_weight;
   HYPRE_Real      one_minus_omega;
   HYPRE_Real      prod;

   one_minus_weight = 1.0 - relax_weight;
   one_minus_omega = 1.0 - omega;
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 0 -> Jacobi or CF-Jacobi
    *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
    *     relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
    *                                     boundary sequential
    *     relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *                               with outer relaxation parameters (forward solve)
    *     relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *                               with outer relaxation parameters (backward solve)
    *     relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
    *     relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
    *                               with outer relaxation parameters
    *     relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
    *     relax_type = 8 -> hybrid L1 Symm. Gauss-Seidel
    *     relax_type = 10 -> On-processor direct forward solve for matrices with
    *                        triangular structure (indices need not be ordered
    *                        triangular)
    *     relax_type = 13 -> hybrid L1 Gauss-Seidel forward solve
    *     relax_type = 14 -> hybrid L1 Gauss-Seidel backward solve
    *     relax_type = 15 -> CG
    *     relax_type = 16 -> Scaled Chebyshev
    *     relax_type = 17 -> FCF-Jacobi
    *     relax_type = 18 -> L1-Jacobi
    *     relax_type = 9, 99, 98 -> Direct solve, Gaussian elimination
    *     relax_type = 19-> Direct Solve, (old version)
    *     relax_type = 29-> Direct solve: use gaussian elimination & BLAS
    *                       (with pivoting) (old version)
    *-----------------------------------------------------------------------*/

   switch (relax_type)
   {
      case 0: /* Weighted Jacobi */
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            /* printf("!! Proc %d: n %d,  num_sends %d, num_cols_offd %d\n", my_id, n, num_sends, num_cols_offd); */

            v_buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               {
                  v_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);
         }
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         if (num_procs > 1)
         {
            hypre_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= one_minus_weight;
                  u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
               }
            }
         }
         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= one_minus_weight;
                  u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
               }
            }
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }
      }
      break;

      case 5: /* Hybrid: Jacobi off-processor,
                         chaotic Gauss-Seidel on-processor       */
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
               {
                  v_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++) /* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }
      }
      break;

      /* Hybrid: Jacobi off-processor, Gauss-Seidel on-processor (forward loop) */
      case 3:
      {
         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

#if defined(HYPRE_USING_PERSISTENT_COMM)
         // JSP: persistent comm can be similarly used for other smoothers
         hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

         if (num_procs > 1)
         {
#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

#if defined(HYPRE_USING_PERSISTENT_COMM)
            persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
            v_buf_data = (HYPRE_Real *) hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
            Vext_data  = (HYPRE_Real *) hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
#else
            v_buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);
#endif

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
            HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
            for (i = begin; i < end; i++)
            {
               v_buf_data[i-begin] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
            }

#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_PERSISTENT_COMM)
            hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, HYPRE_MEMORY_HOST, v_buf_data);
#else
            comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
#endif

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
#if defined(HYPRE_USING_PERSISTENT_COMM)
            hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_HOST, Vext_data);
#else
            hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
            comm_handle = NULL;

#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
         }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
        hypre_profile_times[HYPRE_TIMER_ID_RELAX] -= hypre_MPI_Wtime();
#endif

        if (relax_weight == 1 && omega == 1)
        {
           if (relax_points == 0)
           {
              if (num_threads > 1)
              {
                 tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                 for (i = 0; i < n; i++)
                    tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                 for (j = 0; j < num_threads; j++)
                 {
                    size = n/num_threads;
                    rest = n - size*num_threads;
                    if (j < rest)
                    {
                       ns = j*size+j;
                       ne = (j+1)*size+j+1;
                    }
                    else
                    {
                       ns = j*size+rest;
                       ne = (j+1)*size+rest;
                    }
                    for (i = ns; i < ne; i++) /* interior points first */
                    {
                       /*-----------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                       if ( A_diag_data[A_diag_i[i]] != zero)
                       {
                          res = f_data[i];
                          for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                          {
                             ii = A_diag_j[jj];
                             if (ii >= ns && ii < ne)
                                res -= A_diag_data[jj] * u_data[ii];
                             else
                                res -= A_diag_data[jj] * tmp_data[ii];
                          }
                          for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                          {
                             ii = A_offd_j[jj];
                             res -= A_offd_data[jj] * Vext_data[ii];
                          }
                          u_data[i] = res / A_diag_data[A_diag_i[i]];
                       }
                    }
                 }
              }
              else
              {
                 for (i = 0; i < n; i++) /* interior points first */
                 {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    if ( A_diag_data[A_diag_i[i]] != zero)
                    {
                       res = f_data[i];
                       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                       {
                          ii = A_diag_j[jj];
                          res -= A_diag_data[jj] * u_data[ii];
                       }
                       for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                       {
                          ii = A_offd_j[jj];
                          res -= A_offd_data[jj] * Vext_data[ii];
                       }
                       u_data[i] = res / A_diag_data[A_diag_i[i]];
                    }
                 }
              }
           }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/
           else
           {
              if (num_threads > 1)
              {
                 tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                 for (i = 0; i < n; i++)
                    tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                 for (j = 0; j < num_threads; j++)
                 {
                    size = n/num_threads;
                    rest = n - size*num_threads;
                    if (j < rest)
                    {
                       ns = j*size+j;
                       ne = (j+1)*size+j+1;
                    }
                    else
                    {
                       ns = j*size+rest;
                       ne = (j+1)*size+rest;
                    }
                    for (i = ns; i < ne; i++) /* relax interior points */
                    {
                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                       if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
                       {
                          res = f_data[i];
                          for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                          {
                             ii = A_diag_j[jj];
                             if (ii >= ns && ii < ne)
                                res -= A_diag_data[jj] * u_data[ii];
                             else
                                res -= A_diag_data[jj] * tmp_data[ii];
                          }
                          for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                          {
                             ii = A_offd_j[jj];
                             res -= A_offd_data[jj] * Vext_data[ii];
                          }
                          u_data[i] = res / A_diag_data[A_diag_i[i]];
                       }
                    }
                 }
              }
              else
              {
                 for (i = 0; i < n; i++) /* relax interior points */
                 {
                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
                    if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
                    {
                       res = f_data[i];
                       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                       {
                          ii = A_diag_j[jj];
                          res -= A_diag_data[jj] * u_data[ii];
                       }
                       for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                       {
                          ii = A_offd_j[jj];
                          res -= A_offd_data[jj] * Vext_data[ii];
                       }
                       u_data[i] = res / A_diag_data[A_diag_i[i]];
                    }
                 }
              }
           }
        }
        else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
           {
              Vtemp_data[i] = u_data[i];
           }
           prod = (1.0-relax_weight*omega);
           if (relax_points == 0)
           {
              if (num_threads > 1)
              {
                 tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                 for (i = 0; i < n; i++)
                    tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                 for (j = 0; j < num_threads; j++)
                 {
                    size = n/num_threads;
                    rest = n - size*num_threads;
                    if (j < rest)
                    {
                       ns = j*size+j;
                       ne = (j+1)*size+j+1;
                    }
                    else
                    {
                       ns = j*size+rest;
                       ne = (j+1)*size+rest;
                    }
                    for (i = ns; i < ne; i++) /* interior points first */
                    {
                       /*-----------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                       if ( A_diag_data[A_diag_i[i]] != zero)
                       {
                          res = f_data[i];
                          res0 = 0.0;
                          res2 = 0.0;
                          for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                          {
                             ii = A_diag_j[jj];
                             if (ii >= ns && ii < ne)
                             {
                                res0 -= A_diag_data[jj] * u_data[ii];
                                res2 += A_diag_data[jj] * Vtemp_data[ii];
                             }
                             else
                                res -= A_diag_data[jj] * tmp_data[ii];
                          }
                          for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                          {
                             ii = A_offd_j[jj];
                             res -= A_offd_data[jj] * Vext_data[ii];
                          }
                          u_data[i] *= prod;
                          u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                          /*u_data[i] += omega*(relax_weight*res + res0 +
                            one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                       }
                    }
                 }
              }
              else
              {
                 for (i = 0; i < n; i++) /* interior points first */
                 {
                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
                    if ( A_diag_data[A_diag_i[i]] != zero)
                    {
                       res0 = 0.0;
                       res2 = 0.0;
                       res = f_data[i];
                       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                       {
                          ii = A_diag_j[jj];
                          res0 -= A_diag_data[jj] * u_data[ii];
                          res2 += A_diag_data[jj] * Vtemp_data[ii];
                       }
                       for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                       {
                          ii = A_offd_j[jj];
                          res -= A_offd_data[jj] * Vext_data[ii];
                       }
                       u_data[i] *= prod;
                       u_data[i] += relax_weight*(omega*res + res0 +
                             one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                       /*u_data[i] += omega*(relax_weight*res + res0 +
                         one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                    }
                 }
              }
           }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/
           else
           {
              if (num_threads > 1)
              {
                 tmp_data = Ztemp_data;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                 for (i = 0; i < n; i++)
                    tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                 for (j = 0; j < num_threads; j++)
                 {
                    size = n/num_threads;
                    rest = n - size*num_threads;
                    if (j < rest)
                    {
                       ns = j*size+j;
                       ne = (j+1)*size+j+1;
                    }
                    else
                    {
                       ns = j*size+rest;
                       ne = (j+1)*size+rest;
                    }
                    for (i = ns; i < ne; i++) /* relax interior points */
                    {

                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/

                       if (cf_marker[i] == relax_points
                             && A_diag_data[A_diag_i[i]] != zero)
                       {
                          res0 = 0.0;
                          res2 = 0.0;
                          res = f_data[i];
                          for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                          {
                             ii = A_diag_j[jj];
                             if (ii >= ns && ii < ne)
                             {
                                res0 -= A_diag_data[jj] * u_data[ii];
                                res2 += A_diag_data[jj] * Vtemp_data[ii];
                             }
                             else
                                res -= A_diag_data[jj] * tmp_data[ii];
                          }
                          for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                          {
                             ii = A_offd_j[jj];
                             res -= A_offd_data[jj] * Vext_data[ii];
                          }
                          u_data[i] *= prod;
                          u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                          /*u_data[i] += omega*(relax_weight*res + res0 +
                            one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                       }
                    }
                 }


              }
              else
              {
                 for (i = 0; i < n; i++) /* relax interior points */
                 {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is

                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/

                    if (cf_marker[i] == relax_points
                          && A_diag_data[A_diag_i[i]] != zero)
                    {
                       res = f_data[i];
                       res0 = 0.0;
                       res2 = 0.0;
                       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                       {
                          ii = A_diag_j[jj];
                          res0 -= A_diag_data[jj] * u_data[ii];
                          res2 += A_diag_data[jj] * Vtemp_data[ii];
                       }
                       for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                       {
                          ii = A_offd_j[jj];
                          res -= A_offd_data[jj] * Vext_data[ii];
                       }
                       u_data[i] *= prod;
                       u_data[i] += relax_weight*(omega*res + res0 +
                             one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                       /*u_data[i] += omega*(relax_weight*res + res0 +
                         one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                    }
                 }
              }
           }
        }
#ifndef HYPRE_USING_PERSISTENT_COMM
        if (num_procs > 1)
        {
           hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
           hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
        }
#endif
#ifdef HYPRE_PROFILE
        hypre_profile_times[HYPRE_TIMER_ID_RELAX] += hypre_MPI_Wtime();
#endif
      }
      break;

      case 1: /* Gauss-Seidel VERY SLOW */
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                  hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            status  = hypre_CTAlloc(hypre_MPI_Status, num_recvs+num_sends, HYPRE_MEMORY_HOST);
            requests= hypre_CTAlloc(hypre_MPI_Request,  num_recvs+num_sends, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            /*
               for (i = 0; i < n; i++)
               {
               Vtemp_data[i] = u_data[i];
               } */

         }
         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/
         for (p = 0; p < num_procs; p++)
         {
            jr = 0;
            if (p != my_id)
            {
               for (i = 0; i < num_sends; i++)
               {
                  ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
                  if (ip == p)
                  {
                     vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
                     vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
                     for (j=vec_start; j < vec_start+vec_len; j++)
                        v_buf_data[j] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
                     hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL,
                           ip, 0, comm, &requests[jr++]);
                  }
               }
               hypre_MPI_Waitall(jr,requests,status);
               hypre_MPI_Barrier(comm);
            }
            else
            {
               if (num_procs > 1)
               {
                  for (i = 0; i < num_recvs; i++)
                  {
                     ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
                     vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                     vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
                     hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL,
                           ip, 0, comm, &requests[jr++]);
                  }
                  hypre_MPI_Waitall(jr,requests,status);
               }
               if (relax_points == 0)
               {
                  for (i = 0; i < n; i++)
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
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

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
               if (num_procs > 1)
                  hypre_MPI_Barrier(comm);
            }
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
            hypre_TFree(status, HYPRE_MEMORY_HOST);
            hypre_TFree(requests, HYPRE_MEMORY_HOST);
         }
      }
      break;

      case 2: /* Gauss-Seidel: relax interior points in parallel, boundary
                 sequentially */
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                  hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            status  = hypre_CTAlloc(hypre_MPI_Status, num_recvs+num_sends, HYPRE_MEMORY_HOST);
            requests= hypre_CTAlloc(hypre_MPI_Request,  num_recvs+num_sends, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         /*
            for (i = 0; i < n; i++)
            {
            Vtemp_data[i] = u_data[i];
            } */

         /*-----------------------------------------------------------------
          * Relax interior points first
          *-----------------------------------------------------------------*/
         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ((A_offd_i[i+1]-A_offd_i[i]) == zero &&
                     A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }
         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                     && (A_offd_i[i+1]-A_offd_i[i]) == zero
                     && A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }
         for (p = 0; p < num_procs; p++)
         {
            jr = 0;
            if (p != my_id)
            {
               for (i = 0; i < num_sends; i++)
               {
                  ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
                  if (ip == p)
                  {
                     vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
                     vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
                     for (j=vec_start; j < vec_start+vec_len; j++)
                        v_buf_data[j] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
                     hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL,
                           ip, 0, comm, &requests[jr++]);
                  }
               }
               hypre_MPI_Waitall(jr,requests,status);
               hypre_MPI_Barrier(comm);
            }
            else
            {
               if (num_procs > 1)
               {
                  for (i = 0; i < num_recvs; i++)
                  {
                     ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
                     vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                     vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
                     hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL,
                           ip, 0, comm, &requests[jr++]);
                  }
                  hypre_MPI_Waitall(jr,requests,status);
               }
               if (relax_points == 0)
               {
                  for (i = 0; i < n; i++)
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ((A_offd_i[i+1]-A_offd_i[i]) != zero &&
                           A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
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

                     if (cf_marker[i] == relax_points
                           && (A_offd_i[i+1]-A_offd_i[i]) != zero
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
               if (num_procs > 1)
                  hypre_MPI_Barrier(comm);
            }
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
            hypre_TFree(status, HYPRE_MEMORY_HOST);
            hypre_TFree(requests, HYPRE_MEMORY_HOST);
         }
      }
      break;

      case 4: /* Hybrid: Jacobi off-processor,
                 Gauss-Seidel/SOR on-processor
                 (backward loop) */
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                  hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
                  v_buf_data[index++]
                     = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                  Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                                 res -= A_diag_data[jj] * u_data[ii];
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else
               {
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }

               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                                 res -= A_diag_data[jj] * u_data[ii];
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           res0 = 0.0;
                           res2 = 0.0;
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                  }
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res0 = 0.0;
                        res2 = 0.0;
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,res0,res2,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                  }
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
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
      }
      break;

      case 6: /* Hybrid: Jacobi off-processor,
                 Symm. Gauss-Seidel/ SSOR on-processor
                 with outer relaxation parameter */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                  hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
                  v_buf_data[index++]
                     = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                  Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
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
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] = res / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,res0,res2,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                     }
                  }
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,res0,res2,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
                     }
                  }
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
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
      }
      break;

      case 7: /* Jacobi (uses ParMatvec) */
      {

         /*-----------------------------------------------------------------
          * Copy f into temporary vector.
          *-----------------------------------------------------------------*/
         hypre_SeqVectorPrefetch(hypre_ParVectorLocalVector(Vtemp), HYPRE_MEMORY_DEVICE);
         hypre_SeqVectorPrefetch(hypre_ParVectorLocalVector(f), HYPRE_MEMORY_DEVICE);
         hypre_ParVectorCopy(f, Vtemp);

         /*-----------------------------------------------------------------
          * Perform Matvec Vtemp=f-Au
          *-----------------------------------------------------------------*/

         hypre_ParCSRMatrixMatvec(-relax_weight,A, u, relax_weight, Vtemp);
#if defined(HYPRE_USING_CUDA)
         hypreDevice_IVAXPY(n, l1_norms, Vtemp_data, u_data);
#else
         for (i = 0; i < n; i++)
         {
            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/
            u_data[i] += Vtemp_data[i] / l1_norms[i];
         }
#endif
      }
      break;

      case 8: /* hybrid L1 Symm. Gauss-Seidel */
      {
         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
               {
                  v_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
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
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }
            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }
               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }
               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
                     }
                  }
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
                     }
                  }
               }
            }
            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points && l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
                     }
                  }
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
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
      }
      break;

      /* Hybrid: Jacobi off-processor, ordered Gauss-Seidel on-processor */
      case 10:
      {
         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

#ifdef HYPRE_USING_PERSISTENT_COMM
         // JSP: persistent comm can be similarly used for other smoothers
         hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

         if (num_procs > 1)
         {
#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

#ifdef HYPRE_USING_PERSISTENT_COMM
            persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
            v_buf_data = (HYPRE_Real *) hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
            Vext_data  = (HYPRE_Real *) hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
#else
            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);
#endif

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
            HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
            for (i = begin; i < end; i++)
            {
               v_buf_data[i - begin]
                  = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
            }

#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_USING_PERSISTENT_COMM
            hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, HYPRE_MEMORY_HOST, v_buf_data);
#else
            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);
#endif

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_PERSISTENT_COMM
            hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_HOST, Vext_data);
#else
            hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
            comm_handle = NULL;

#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
         }

         // Check for ordering of matrix. If stored, get pointer, otherwise
         // compute ordering and point matrix variable to array.
         HYPRE_Int *proc_ordering;
         if (!hypre_ParCSRMatrixProcOrdering(A)) {
            proc_ordering = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
            hypre_topo_sort(A_diag_i, A_diag_j, A_diag_data, proc_ordering, n);
            hypre_ParCSRMatrixProcOrdering(A) = proc_ordering;
         }
         else {
            proc_ordering = hypre_ParCSRMatrixProcOrdering(A);
         }

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
        hypre_profile_times[HYPRE_TIMER_ID_RELAX] -= hypre_MPI_Wtime();
#endif

         if (relax_points == 0)
         {
            if (num_threads > 1)
            {
               tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
                  tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
               for (j = 0; j < num_threads; j++)
               {
                  size = n/num_threads;
                  rest = n - size*num_threads;
                  if (j < rest)
                  {
                     ns = j*size+j;
                     ne = (j+1)*size+j+1;
                  }
                  else
                  {
                     ns = j*size+rest;
                     ne = (j+1)*size+rest;
                  }
                  for (i = ns; i < ne; i++)   /* interior points first */
                  {
                     HYPRE_Int row = proc_ordering[i];
                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point row; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     if ( A_diag_data[A_diag_i[row]] != zero)
                     {
                        res = f_data[row];
                        for (jj = A_diag_i[row]+1; jj < A_diag_i[row+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                              res -= A_diag_data[jj] * u_data[ii];
                           else
                              res -= A_diag_data[jj] * tmp_data[ii];
                        }
                        for (jj = A_offd_i[row]; jj < A_offd_i[row+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[row] = res / A_diag_data[A_diag_i[row]];
                     }
                  }
               }
            }
            else
            {
               for (i = 0; i < n; i++) /* interior points first */
               {
                  HYPRE_Int row = proc_ordering[i];
                  /*-----------------------------------------------------------
                   * If diagonal is nonzero, relax point i; otherwise, skip it.
                   *-----------------------------------------------------------*/
                  if ( A_diag_data[A_diag_i[row]] != zero)
                  {
                     res = f_data[row];
                     for (jj = A_diag_i[row]+1; jj < A_diag_i[row+1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     for (jj = A_offd_i[row]; jj < A_offd_i[row+1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        res -= A_offd_data[jj] * Vext_data[ii];
                     }
                     u_data[row] = res / A_diag_data[A_diag_i[row]];
                  }
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/
         else
         {
            if (num_threads > 1)
            {
               tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < n; i++)
                  tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
               for (j = 0; j < num_threads; j++)
               {
                  size = n/num_threads;
                  rest = n - size*num_threads;
                  if (j < rest)
                  {
                     ns = j*size+j;
                     ne = (j+1)*size+j+1;
                  }
                  else
                  {
                     ns = j*size+rest;
                     ne = (j+1)*size+rest;
                  }
                  for (i = ns; i < ne; i++) /* relax interior points */
                  {
                     HYPRE_Int row = proc_ordering[i];
                     /*-----------------------------------------------------------
                      * If row is of the right type ( C or F ) and diagonal is
                      * nonzero, relax point row; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     if (cf_marker[row] == relax_points
                         && A_diag_data[A_diag_i[row]] != zero)
                     {
                        res = f_data[row];
                        for (jj = A_diag_i[row]+1; jj < A_diag_i[row+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                              res -= A_diag_data[jj] * u_data[ii];
                           else
                              res -= A_diag_data[jj] * tmp_data[ii];
                        }
                        for (jj = A_offd_i[row]; jj < A_offd_i[row+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[row] = res / A_diag_data[A_diag_i[row]];
                     }
                  }
               }
            }
            else
            {
               for (i = 0; i < n; i++) /* relax interior points */
               {
                  HYPRE_Int row = proc_ordering[i];
                  /*-----------------------------------------------------------
                   * If row is of the right type ( C or F ) and diagonal is
                   * nonzero, relax point row; otherwise, skip it.
                   *-----------------------------------------------------------*/
                  if (cf_marker[row] == relax_points
                      && A_diag_data[A_diag_i[row]] != zero)
                  {
                     res = f_data[row];
                     for (jj = A_diag_i[row]+1; jj < A_diag_i[row+1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     for (jj = A_offd_i[row]; jj < A_offd_i[row+1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        res -= A_offd_data[jj] * Vext_data[ii];
                     }
                     u_data[row] = res / A_diag_data[A_diag_i[row]];
                  }
               }
            }
         }

#ifndef HYPRE_USING_PERSISTENT_COMM
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
            hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
         }
#endif
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_RELAX] += hypre_MPI_Wtime();
#endif
      }
      break;

      case 13: /* hybrid L1 Gauss-Seidel forward solve */
      {
         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real,
                  hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
                  v_buf_data[index++]
                     = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
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
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }
               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
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
      }
      break;

      case 14: /* hybrid L1 Gauss-Seidel backward solve */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

            v_buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

            Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
               {
                  v_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }

            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
            hypre_ParCSRCommHandleDestroy(comm_handle);
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
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] += res / l1_norms[i];
                        }
                     }
                  }

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] += res / l1_norms[i];
                     }
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < n; i++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* interior points first */
                     {

                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if ( l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* interior points first */
                  {

                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if ( l1_norms[i] != zero)
                     {
                        res0 = 0.0;
                        res = f_data[i];
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ne-1; i > ns-1; i--) /* relax interior points */
                     {

                        /*-----------------------------------------------------------
                         * If i is of the right type ( C or F ) and diagonal is
                         * nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/

                        if (cf_marker[i] == relax_points
                              && l1_norms[i] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                                 res0 -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                                 res -= A_diag_data[jj] * tmp_data[ii];
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                                 one_minus_omega*res2) / l1_norms[i];
                           /*u_data[i] += omega*(relax_weight*res + res0 +
                             one_minus_weight*res2) / l1_norms[i];*/
                        }
                     }
                  }

               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is

                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/

                     if (cf_marker[i] == relax_points
                           && l1_norms[i] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                              one_minus_omega*res2) / l1_norms[i];
                        /*u_data[i] += omega*(relax_weight*res + res0 +
                          one_minus_weight*res2) / l1_norms[i];*/
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
      }
      break;

      case 19: /* Direct solve: use gaussian elimination */
      {
         HYPRE_Int n_global = (HYPRE_Int) global_num_rows;
         HYPRE_Int first_index = (HYPRE_Int) first_ind;
         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = hypre_ParVectorToVectorAll(f);
#endif
         if (n)
         {
#ifndef HYPRE_NO_GLOBAL_PARTITION
            A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
            f_vector = hypre_ParVectorToVectorAll(f);
#endif
            A_CSR_i = hypre_CSRMatrixI(A_CSR);
            A_CSR_j = hypre_CSRMatrixJ(A_CSR);
            A_CSR_data = hypre_CSRMatrixData(A_CSR);
            f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real, n_global*n_global, HYPRE_MEMORY_HOST);
            b_vec = hypre_CTAlloc(HYPRE_Real, n_global, HYPRE_MEMORY_HOST);

            /*---------------------------------------------------------------
             *  Load CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i*n_global+column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            hypre_gselim(A_mat,b_vec,n_global,relax_error);

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index+i];
            }

            hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
            hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#ifdef HYPRE_NO_GLOBAL_PARTITION
         else
         {

            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

      }
      break;
      case 98: /* Direct solve: use gaussian elimination & BLAS (with pivoting) */
      {

         HYPRE_Int n_global = (HYPRE_Int) global_num_rows;
         HYPRE_Int first_index = (HYPRE_Int) first_ind;
         HYPRE_Int info;
         HYPRE_Int one_i = 1;
         HYPRE_Int *piv;
         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = hypre_ParVectorToVectorAll(f);
#endif
         if (n)
         {
#ifndef HYPRE_NO_GLOBAL_PARTITION
            A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
            f_vector = hypre_ParVectorToVectorAll(f);
#endif
            A_CSR_i = hypre_CSRMatrixI(A_CSR);
            A_CSR_j = hypre_CSRMatrixJ(A_CSR);
            A_CSR_data = hypre_CSRMatrixData(A_CSR);
            f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real,  n_global*n_global, HYPRE_MEMORY_HOST);
            b_vec = hypre_CTAlloc(HYPRE_Real,  n_global, HYPRE_MEMORY_HOST);

            /*---------------------------------------------------------------
             *  Load CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
               {

                  /* need col major */
                  column = A_CSR_j[jj];
                  A_mat[i + n_global*column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            piv = hypre_CTAlloc(HYPRE_Int,  n_global, HYPRE_MEMORY_HOST);

            /* write over A with LU */
            hypre_dgetrf(&n_global, &n_global, A_mat, &n_global, piv, &info);

            /*now b_vec = inv(A)*b_vec  */
            hypre_dgetrs("N", &n_global, &one_i, A_mat, &n_global, piv, b_vec, &n_global, &info);

            hypre_TFree(piv, HYPRE_MEMORY_HOST);

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index+i];
            }

            hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
            hypre_TFree(b_vec, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#ifdef HYPRE_NO_GLOBAL_PARTITION
         else
         {

            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif
      }
      break;
   }

   return (relax_error);
}


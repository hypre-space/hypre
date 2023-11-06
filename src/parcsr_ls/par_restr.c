/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

#define AIR_DEBUG 0
#define EPSILON 1e-18
#define EPSIMAC 1e-16

void hypre_fgmresT(HYPRE_Int n, HYPRE_Complex *A, HYPRE_Complex *b, HYPRE_Real tol, HYPRE_Int kdim,
                   HYPRE_Complex *x, HYPRE_Real *relres, HYPRE_Int *iter, HYPRE_Int job);
void hypre_ordered_GS(const HYPRE_Complex L[], const HYPRE_Complex rhs[], HYPRE_Complex x[],
                      const HYPRE_Int n);

HYPRE_Int
hypre_BoomerAMGBuildRestrAIR( hypre_ParCSRMatrix   *A,
                              HYPRE_Int            *CF_marker,
                              hypre_ParCSRMatrix   *S,
                              HYPRE_BigInt         *num_cpts_global,
                              HYPRE_Int             num_functions,
                              HYPRE_Int            *dof_func,
                              HYPRE_Real            filter_thresholdR,
                              HYPRE_Int             debug_flag,
                              hypre_ParCSRMatrix  **R_ptr,
                              HYPRE_Int             is_triangular,
                              HYPRE_Int             gmres_switch)
{
   HYPRE_UNUSED_VAR(debug_flag);

   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   /* diag part of A */
   hypre_CSRMatrix *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i    = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j    = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd      = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i    = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j    = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   hypre_ParCSRMatrix *R;
   /* csr's */
   hypre_CSRMatrix *R_diag;
   hypre_CSRMatrix *R_offd;
   /* arrays */
   HYPRE_Complex      *R_diag_data;
   HYPRE_Int       *R_diag_i;
   HYPRE_Int       *R_diag_j;
   HYPRE_Complex      *R_offd_data;
   HYPRE_Int       *R_offd_i;
   HYPRE_Int       *R_offd_j;
   HYPRE_BigInt    *col_map_offd_R = NULL;
   HYPRE_Int       *tmp_map_offd = NULL;
   /* CF marker off-diag part */
   HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   HYPRE_Int       *dof_func_offd  = NULL;
   /* ghost rows */
   hypre_CSRMatrix *A_ext      = NULL;
   HYPRE_Complex      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i    = NULL;
   HYPRE_BigInt    *A_ext_j    = NULL;

   HYPRE_Int        i, j, k, i1, k1, k2, rr, cc, ic, index, start,
                    local_max_size, local_size, num_cols_offd_R;

   /* LAPACK */
   HYPRE_Complex *DAi, *Dbi, *Dxi;
#if AIR_DEBUG
   HYPRE_Complex *TMPA, *TMPb, *TMPd;
#endif
   HYPRE_Int *Ipi, lapack_info, ione = 1;
   char charT = 'T';
   char Aisol_method;

   /* if the size of local system is larger than gmres_switch, use GMRES */
   HYPRE_Int gmresAi_maxit = 50;
   HYPRE_Real gmresAi_tol = 1e-3;

   HYPRE_Int my_id, num_procs;
   HYPRE_BigInt total_global_cpts/*, my_first_cpt*/;
   HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_Int *marker_diag, *marker_offd;
   HYPRE_Int num_sends, *int_buf_data;
   /* local size, local num of C points */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int n_cpts = 0;
   /* my first column range */
   HYPRE_BigInt col_start = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt col_end   = col_start + (HYPRE_BigInt)n_fine;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   /* number of sends to do (number of procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);
   /* copy CF markers of elements to send to buffer
    * RL: why copy them with two for loops? Why not just loop through all in one */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   /* destroy the handle to finish communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);
   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine the nnz of R and the max local size
    *-----------------------------------------------------------------------*/
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;
   for (i = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }
      /* local number of C-pts */
      n_cpts ++;
      /* If i is a C-point, the restriction is from the F-points that
       * strongly influence i */
      local_size = 0;
      /* loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            cnt_diag ++;
            local_size ++;
         }
      }
      /* if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         /* use this mapping to have offd indices of A */
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            i1 = S_offd_j[j];
            if (CF_marker_offd[i1] < 0)
            {
               cnt_offd ++;
               local_size ++;
            }
         }
      }
      /* keep ths max size */
      local_max_size = hypre_max(local_max_size, local_size);
   }

   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;

   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   R_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, HYPRE_MEMORY_HOST);
   R_diag_j    = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   R_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_diag, HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, HYPRE_MEMORY_HOST);
   R_offd_j    = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   R_offd_data = hypre_CTAlloc(HYPRE_Complex, nnz_offd, HYPRE_MEMORY_HOST);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   /*----------------------------------------       .-.
    * Get the GHOST rows of A,                     (o o) boo!
    * i.e., adjacent rows to this proc             | O \
    * whose row indices are in A->col_map_offd      \   \
    *-----------------------------------------       `~~~'  */
   /* external rows of A that are needed for perform A multiplication,
    * the last arg means need data
    * the number of rows is num_cols_A_offd */
   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
   }

   /* marker array: if this point is i's strong F neighbors
    *             >=  0: yes, and is the local dense id
    *             == -1: no */
   marker_diag = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      marker_diag[i] = -1;
   }
   marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd; i++)
   {
      marker_offd[i] = -1;
   }

   // Allocate the rhs and dense local matrix in column-major form (for LAPACK)
   DAi = hypre_CTAlloc(HYPRE_Complex, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   Dbi = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   Dxi = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   Ipi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST); // pivot matrix

   // Allocate memory for GMRES if it will be used
   HYPRE_Int kdim_max = hypre_min(gmresAi_maxit, local_max_size);
   if (gmres_switch < local_max_size)
   {
      hypre_fgmresT(local_max_size, NULL, NULL, 0.0, kdim_max, NULL, NULL, NULL, -1);
   }

#if AIR_DEBUG
   /* FOR DEBUG */
   TMPA = hypre_CTAlloc(HYPRE_Complex, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   TMPb = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   TMPd = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
#endif

   /*-----------------------------------------------------------------------
    *  Second Pass: Populate R
    *-----------------------------------------------------------------------*/
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of Ai, bi */
      local_size = 0;

      /* If i is a C-point, build the restriction, from the F-points that
       * strongly influence i
       * Access S for the first time, mark the points we want */
      /* 1: loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] == -1);
            /* mark this point */
            marker_diag[i1] = local_size ++;
         }
      }
      /* 2: if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] == -1);
               /* mark this point */
               marker_offd[i1] = local_size ++;
            }
         }
      }

      /* DEBUG FOR local_size == 0 */
      /*
      if (local_size == 0)
      {
         printf("my_id %d:  ", my_id);
         for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
         {
            i1 = S_diag_j[j];
            printf("%d[d, %d] ", i1, CF_marker[i1]);
         }
         printf("\n");
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            i1 = S_offd_j[j];
            printf("%d[o, %d] ", i1, CF_marker_offd[i1]);
         }

         printf("\n");

         exit(0);
      }
      */

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(HYPRE_Complex));
      memset(Dxi, 0, local_size * sizeof(HYPRE_Complex));
      memset(Dbi, 0, local_size * sizeof(HYPRE_Complex));

      /* we will populate Ai, bi row-by-row
       * rr is the local dense matrix row counter */
      rr = 0;
      /* 1. diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         /* row i1 */
         i1 = S_diag_j[j];
         /* i1 is an F point */
         if (CF_marker[i1] < 0)
         {
            /* go through row i1 of A: a local row */
            /* diag part of row i1 */
            for (k = A_diag_i[i1]; k < A_diag_i[i1 + 1]; k++)
            {
               k1 = A_diag_j[k];
               /* if this col is marked with its local dense id */
               if ((cc = marker_diag[k1]) >= 0)
               {
                  hypre_assert(CF_marker[k1] < 0);
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_data[k];
               }
            }
            /* if parallel, offd part of row i1 */
            if (num_procs > 1)
            {
               for (k = A_offd_i[i1]; k < A_offd_i[i1 + 1]; k++)
               {
                  k1 = A_offd_j[k];
                  /* if this col is marked with its local dense id */
                  if ((cc = marker_offd[k1]) >= 0)
                  {
                     hypre_assert(CF_marker_offd[k1] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = A_offd_data[k];
                  }
               }
            }
            /* done with row i1 */
            rr++;
         }
      } /* for (j=...), diag part of row i done */

      /* 2. if parallel, offd part of row i. The corresponding rows are
       *    in matrix A_ext */
      if (num_procs > 1)
      {
         HYPRE_BigInt big_k1;
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* row i1: use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* if this is an F point */
            if (CF_marker_offd[i1] < 0)
            {
               /* loop through row i1 of A_ext, a global CSR matrix */
               for (k = A_ext_i[i1]; k < A_ext_i[i1 + 1]; k++)
               {
                  /* k1 is a global index! */
                  big_k1 = A_ext_j[k];
                  if (big_k1 >= col_start && big_k1 < col_end)
                  {
                     /* big_k1 is in the diag part, adjust to local index */
                     k1 = (HYPRE_Int)(big_k1 - col_start);
                     /* if this col is marked with its local dense id*/
                     if ((cc = marker_diag[k1]) >= 0)
                     {
                        hypre_assert(CF_marker[k1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = A_ext_data[k];
                     }
                  }
                  else
                  {
                     /* k1 is in the offd part
                      * search k1 in A->col_map_offd */
                     k2 = hypre_BigBinarySearch(col_map_offd_A, big_k1, num_cols_A_offd);
                     /* if found, k2 is the position of column id k1 in col_map_offd */
                     if (k2 > -1)
                     {
                        /* if this col is marked with its local dense id */
                        if ((cc = marker_offd[k2]) >= 0)
                        {
                           hypre_assert(CF_marker_offd[k2] < 0);
                           /* copy the value */
                           /* rr and cc: local dense ids */
                           DAi[rr + cc * local_size] = A_ext_data[k];
                        }
                     }
                  }
               }
               /* done with row i1 */
               rr++;
            }
         }
      }

      hypre_assert(rr == local_size);

      /* assemble rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = marker_diag[i1]) >= 0)
         {
            /* this should be true but not very important
             * what does it say is that eqn order == unknown order
             * this is true, since order in A is preserved in S */
            hypre_assert(rr == cc);
            /* Note the sign change */
            Dbi[cc] = -A_diag_data[j];
            rr++;
         }
      }
      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            i1 = A_offd_j[j];
            if ((cc = marker_offd[i1]) >= 0)
            {
               /* this should be true but not very important
                * what does it say is that eqn order == unknown order
                * this is true, since order in A is preserved in S */
               hypre_assert(rr == cc);
               /* Note the sign change */
               Dbi[cc] = -A_offd_data[j];
               rr++;
            }
         }
      }
      hypre_assert(rr == local_size);

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * We have Ai and bi built. Solve the linear system by:
       *    - forward solve for triangular matrix
       *    - LU factorization (LAPACK) for local_size <= gmres_switch
       *    - Dense GMRES for local_size > gmres_switch
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      Aisol_method = local_size <= gmres_switch ? 'L' : 'G';
      if (local_size > 0)
      {
         if (is_triangular)
         {
            hypre_ordered_GS(DAi, Dbi, Dxi, local_size);
#if AIR_DEBUG
            HYPRE_Real alp = -1.0, err;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               hypre_printf("triangular solve res: %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve using LAPACK and LU factorization
         else if (Aisol_method == 'L')
         {
#if AIR_DEBUG
            memcpy(TMPA, DAi, local_size * local_size * sizeof(HYPRE_Complex));
            memcpy(TMPb, Dbi, local_size * sizeof(HYPRE_Complex));
#endif
            lapack_info = 0;
            hypre_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi,
                         &lapack_info);

            hypre_assert(lapack_info == 0);

            if (lapack_info == 0)
            {
               /* solve A_i^T x_i = b_i,
                * solution is saved in b_i on return */
               hypre_dgetrs(&charT, &local_size, &ione, DAi, &local_size,
                            Ipi, Dbi, &local_size, &lapack_info);
               hypre_assert(lapack_info == 0);
            }
#if AIR_DEBUG
            HYPRE_Real alp = 1.0, bet = 0.0, err;
            hypre_dgemv(&charT, &local_size, &local_size, &alp, TMPA, &local_size, Dbi,
                        &ione, &bet, TMPd, &ione);
            alp = -1.0;
            hypre_daxpy(&local_size, &alp, TMPb, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               hypre_printf("dense: local res norm %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve by GMRES
         else
         {
            HYPRE_Real gmresAi_res;
            HYPRE_Int  gmresAi_niter;
            HYPRE_Int kdim = hypre_min(gmresAi_maxit, local_size);

            hypre_fgmresT(local_size, DAi, Dbi, gmresAi_tol, kdim, Dxi,
                          &gmresAi_res, &gmresAi_niter, 0);

            if (gmresAi_res > gmresAi_tol)
            {
               hypre_printf("gmres/jacobi not converge to %e: final_res %e\n", gmresAi_tol, gmresAi_res);
            }

#if AIR_DEBUG
            HYPRE_Real err, nrmb;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            HYPRE_Real alp = -1.0;
            nrmb = hypre_dnrm2(&local_size, Dbi, &ione);
            hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err / nrmb > gmresAi_tol)
            {
               hypre_printf("GMRES/Jacobi: res norm %e, nrmb %e, relative %e\n", err, nrmb, err / nrmb);
               hypre_printf("GMRES/Jacobi: relative %e\n", gmresAi_res);
               exit(0);
            }
#endif
         }
      }

      HYPRE_Complex *Soli = (is_triangular || (Aisol_method == 'G')) ? Dxi : Dbi;

      /* now we are ready to fill this row of R */
      /* diag part */
      rr = 0;
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] == rr);
            /* col idx: use i1, local idx  */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Soli[rr++];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i''; */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic + 1] = cnt_diag;

      /* offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] == rr);
               /* col idx: use the local col id of A_offd,
                * and you will see why later (very soon!) */
               R_offd_j[cnt_offd] = i1;
               /* copy the value */
               R_offd_data[cnt_offd++] = Soli[rr++];
            }
         }
      }
      /* row ptr of the next row */
      R_offd_i[ic + 1] = cnt_offd;

      /* we must have copied all entries */
      hypre_assert(rr == local_size);

      /* reset markers */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            hypre_assert(marker_diag[i1] >= 0);
            marker_diag[i1] = -1;
         }
      }
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               hypre_assert(marker_offd[i1] >= 0);
               marker_offd[i1] = -1;
            }
         }
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   hypre_assert(ic == n_cpts);
   hypre_assert(cnt_diag == nnz_diag);
   hypre_assert(cnt_offd == nnz_offd);

   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, marker_offd should be all -1 */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (marker_offd[i1] == -1)
      {
         num_cols_offd_R++;
         marker_offd[i1] = 1;
      }
   }

   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the offd-idx of A */
   if (num_cols_offd_R)
   {
      col_map_offd_R = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_R, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_R, HYPRE_MEMORY_HOST);
   }
   for (i = 0, i1 = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i] == 1)
      {
         tmp_map_offd[i1++] = i;
      }
   }
   hypre_assert(i1 == num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t col_map_offd_R
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = hypre_BinarySearch(tmp_map_offd, i1, num_cols_offd_R);
      /* search must succeed */
      hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = col_map_offd_A[tmp_map_offd[i]];
   }

   /* Now, we should have everything of Parcsr matrix R */
   R = hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixData(R_diag) = R_diag_data;
   hypre_CSRMatrixI(R_diag)    = R_diag_i;
   hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixData(R_offd) = R_offd_data;
   hypre_CSRMatrixI(R_offd)    = R_offd_i;
   hypre_CSRMatrixJ(R_offd)    = R_offd_j;

   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_ParCSRMatrixAssumedPartition(R) = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   /* free workspace */
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(DAi, HYPRE_MEMORY_HOST);
   hypre_TFree(Dbi, HYPRE_MEMORY_HOST);
   hypre_TFree(Dxi, HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   hypre_TFree(TMPA, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPb, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPd, HYPRE_MEMORY_HOST);
#endif
   hypre_TFree(Ipi, HYPRE_MEMORY_HOST);
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(A_ext);
   }

   if (gmres_switch < local_max_size)
   {
      hypre_fgmresT(0, NULL, NULL, 0.0, 0, NULL, NULL, NULL, -2);
   }

   return 0;
}

/* Compute matvec A^Tx = y, where A is stored in column major form. */
// This can also probably be accomplished with BLAS
static inline void
colmaj_mvT(HYPRE_Complex *A,
           HYPRE_Complex *x,
           HYPRE_Complex *y,
           HYPRE_Int      n)
{
   memset(y, 0, n * sizeof(HYPRE_Complex));
   HYPRE_Int i, j;
   for (i = 0; i < n; i++)
   {
      HYPRE_Int row0 = i * n;
      for (j = 0; j < n; j++)
      {
         y[i] += x[j] * A[row0 + j];
      }
   }
}

// TODO : need to initialize and de-initialize GMRES
void
hypre_fgmresT(HYPRE_Int      n,
              HYPRE_Complex *A,
              HYPRE_Complex *b,
              HYPRE_Real     tol,
              HYPRE_Int      kdim,
              HYPRE_Complex *x,
              HYPRE_Real    *relres,
              HYPRE_Int     *iter,
              HYPRE_Int      job)
{
   HYPRE_Int one = 1, i, j, k;
   static HYPRE_Complex *V = NULL, *Z = NULL, *H = NULL, *c = NULL, *s = NULL, *rs = NULL;
   HYPRE_Complex *v, *z, *w;
   HYPRE_Real t, normr, normr0, tolr;

   if (job == -1)
   {
      V  = hypre_TAlloc(HYPRE_Complex, n * (kdim + 1),    HYPRE_MEMORY_HOST);
      /* Z  = hypre_TAlloc(HYPRE_Complex, n*kdim,        HYPRE_MEMORY_HOST); */
      /* XXX NO PRECOND */
      Z = V;
      H  = hypre_TAlloc(HYPRE_Complex, (kdim + 1) * kdim, HYPRE_MEMORY_HOST);
      c  = hypre_TAlloc(HYPRE_Complex, kdim,          HYPRE_MEMORY_HOST);
      s  = hypre_TAlloc(HYPRE_Complex, kdim,          HYPRE_MEMORY_HOST);
      rs = hypre_TAlloc(HYPRE_Complex, kdim + 1,        HYPRE_MEMORY_HOST);
      return;
   }
   else if (job == -2)
   {
      hypre_TFree(V,  HYPRE_MEMORY_HOST);
      /* hypre_TFree(Z,  HYPRE_MEMORY_HOST); */
      Z = NULL;
      hypre_TFree(H,  HYPRE_MEMORY_HOST);
      hypre_TFree(c,  HYPRE_MEMORY_HOST);
      hypre_TFree(s,  HYPRE_MEMORY_HOST);
      hypre_TFree(rs, HYPRE_MEMORY_HOST);
      return;
   }

   /* XXX: x_0 is all ZERO !!! so r0 = b */
   v = V;
   hypre_TMemcpy(v, b, HYPRE_Complex, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   normr = normr0 = hypre_sqrt(hypre_ddot(&n, v, &one, v, &one));

   if (normr0 < EPSIMAC)
   {
      return;
   }

   tolr = tol * normr0;

   rs[0] = normr0;
   t = 1.0 / normr0;
   hypre_dscal(&n, &t, v, &one);
   i = 0;
   while (i < kdim)
   {
      i++;
      // zi = M^{-1} * vi;
      v = V + (i - 1) * n;
      z = Z + (i - 1) * n;
      /* XXX NO PRECOND */
      /* memcpy(z, v, n*sizeof(HYPRE_Complex)); */
      // w = v_{i+1} = A * zi
      w = V + i * n;
      colmaj_mvT(A, z, w, n);
      // modified Gram-schmidt
      for (j = 0; j < i; j++)
      {
         v = V + j * n;
         H[j + (i - 1)*kdim] = t = hypre_ddot(&n, v, &one, w, &one);
         t = -t;
         hypre_daxpy(&n, &t, v, &one, w, &one);
      }
      H[i + (i - 1)*kdim] = t = hypre_sqrt(hypre_ddot(&n, w, &one, w, &one));
      if (hypre_abs(t) > EPSILON)
      {
         t = 1.0 / t;
         hypre_dscal(&n, &t, w, &one);
      }
      // Least square problem of H
      for (j = 1; j < i; j++)
      {
         t = H[j - 1 + (i - 1) * kdim];
         H[j - 1 + (i - 1)*kdim] =  c[j - 1] * t + s[j - 1] * H[j + (i - 1) * kdim];
         H[j + (i - 1)*kdim]   = -s[j - 1] * t + c[j - 1] * H[j + (i - 1) * kdim];
      }
      HYPRE_Complex hii  = H[i - 1 + (i - 1) * kdim];
      HYPRE_Complex hii1 = H[i + (i - 1) * kdim];
      HYPRE_Complex gam = hypre_sqrt(hii * hii + hii1 * hii1);

      if (hypre_cabs(gam) < EPSILON)
      {
         gam = EPSIMAC;
      }
      c[i - 1] = hii / gam;
      s[i - 1] = hii1 / gam;
      rs[i]   = -s[i - 1] * rs[i - 1];
      rs[i - 1] =  c[i - 1] * rs[i - 1];
      // residue norm
      H[i - 1 + (i - 1)*kdim] = c[i - 1] * hii + s[i - 1] * hii1;
      normr = hypre_cabs(rs[i]);
      if (normr <= tolr)
      {
         break;
      }
   }

   // solve the upper triangular system
   rs[i - 1] /= H[i - 1 + (i - 1) * kdim];
   for (k = i - 2; k >= 0; k--)
   {
      for (j = k + 1; j < i; j++)
      {
         rs[k] -= H[k + j * kdim] * rs[j];
      }
      rs[k] /= H[k + k * kdim];
   }

   // get solution
   for (j = 0; j < i; j++)
   {
      z = Z + j * n;
      hypre_daxpy(&n, rs + j, z, &one, x, &one);
   }

   *relres = normr / normr0;
   *iter = i;
}

/* Ordered Gauss Seidel on A^T in column major format. Since we are
 * solving A^T, equivalent to solving A in row major format. */
void
hypre_ordered_GS(const HYPRE_Complex L[],
                 const HYPRE_Complex rhs[],
                 HYPRE_Complex       x[],
                 const HYPRE_Int     n)
{
   // Get triangular ordering of L^T in col major as ordering of L in row major
   HYPRE_Int *ordering = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   hypre_dense_topo_sort(L, ordering, n, 0);

   // Ordered Gauss-Seidel iteration
   HYPRE_Int i, col;
   for (i = 0; i < n; i++)
   {
      HYPRE_Int row = ordering[i];
      HYPRE_Complex temp = rhs[row];
      for (col = 0; col < n; col++)
      {
         if (col != row)
         {
            temp -= L[row * n + col] * x[col]; // row-major
         }
      }

      HYPRE_Complex diag = L[row * n + row];
      if (hypre_cabs(diag) < 1e-12)
      {
         x[row] = 0.0;
      }
      else
      {
         x[row] = temp / diag;
      }
   }

   hypre_TFree(ordering, HYPRE_MEMORY_HOST);
}

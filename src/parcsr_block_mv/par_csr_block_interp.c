/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_block_mv.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterp

 This is the block version of classical R-S interpolation. We use the complete
 blocks of A (not just the diagonals of these blocks).

 A and P are now Block matrices.  The Strength matrix S is not as it gives
 nodal strengths.

 CF_marker is size number of nodes.

 add_weak_to_diag  0 = don't add weak connections to diag (distribute instead)
 1 = do add

 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix  *A,
                                 HYPRE_Int                *CF_marker,
                                 hypre_ParCSRMatrix       *S,
                                 HYPRE_BigInt             *num_cpts_global,
                                 HYPRE_Int                 num_functions,
                                 HYPRE_Int                *dof_func,
                                 HYPRE_Int                 debug_flag,
                                 HYPRE_Real                trunc_factor,
                                 HYPRE_Int                 max_elmts,
                                 HYPRE_Int                 add_weak_to_diag,
                                 hypre_ParCSRBlockMatrix **P_ptr )
{
   HYPRE_UNUSED_VAR(dof_func);

   MPI_Comm                 comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real           *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int            *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int            *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   HYPRE_Int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int             bnnz = block_size * block_size;

   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real           *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int            *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int            *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   HYPRE_Int             num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_BigInt         *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix      *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int            *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int            *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix      *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int            *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int            *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   HYPRE_BigInt          *col_map_offd_P;
   HYPRE_Int             *tmp_map_offd = NULL;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRBlockMatrix  *A_ext = NULL;
   HYPRE_Real            *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_BigInt          *A_ext_j = NULL;

   hypre_CSRBlockMatrix  *P_diag;
   hypre_CSRBlockMatrix  *P_offd;

   HYPRE_Real            *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Real            *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;

   HYPRE_Int             *P_marker, *P_marker_offd;

   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd = NULL;
   HYPRE_Int              jj_begin_row, jj_begin_row_offd;
   HYPRE_Int              jj_end_row, jj_end_row_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_BigInt          *fine_to_coarse_offd = NULL;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_BigInt           total_global_cpts, my_first_cpt;
   HYPRE_Int              num_cols_P_offd;

   HYPRE_Int              bd;

   HYPRE_Int              i, i1, i2;
   HYPRE_Int              j, jl, jj, jj1;
   HYPRE_Int              kc;
   HYPRE_BigInt           big_k;
   HYPRE_Int              start;

   HYPRE_Int              c_num;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data = NULL;
   HYPRE_BigInt          *big_buf_data = NULL;

   HYPRE_BigInt col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   HYPRE_Int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_BigInt col_n = col_1 + (HYPRE_BigInt)local_numrows;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block;
   HYPRE_Real       *distribute_block;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   /* num_threads = hypre_NumThreads(); */
   num_threads = 1;

   if (num_functions > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for num_functions > 1!");
   }

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);


   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                              CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixBigJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            /* for the data field we must get all of the block data */
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (HYPRE_BigInt)(-kc - 1);
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
               }
               index++;
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/


   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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


      /* loop over the fine grid points */
      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector (fine_to_coarse is the mapping vector).
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   zero_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }


   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   distribute_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);

   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         big_buf_data[index++]
            = my_first_cpt + fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */

   comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data,
                                               fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }

      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      strong_f_marker = -2;

      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            /* P_diag_data[jj_counter] = one; */
            hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                              &P_diag_data[jj_counter * bnnz],
                                              1.0, block_size);
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     /* P_offd_data[jj_counter_offd] = zero; */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);

                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;


            /* get the diagonal block */
            /* diagonal = A_diag_data[A_diag_i[i]]; */
            hypre_CSRBlockMatrixBlockCopyData(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block,
                                              1.0, block_size);



            /* Here we go through the neighborhood of this grid point */

            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         &P_diag_data[P_marker[i1]*bnnz],
                                                         block_size);

               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker || (!add_weak_to_diag  && CF_marker[i1] != -3))
               {
                  /* initialize sum to zero */
                  /* sum = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0,
                                                    block_size);


                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *-----------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row)
                     {
                        /* add diag data to sum */
                        /* sum += A_diag_data[jj1]; */
                        hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj1 * bnnz],
                                                               sum_block, block_size);
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd )
                        {
                           /* add off diag data to sum */
                           /*sum += A_offd_data[jj1];*/
                           hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj1 * bnnz],
                                                                  sum_block, block_size);

                        }
                     }
                  }
                  /* check whether sum_block is singular */

                  /* distribute = A_diag_data[jj] / sum;*/
                  /* here we want: A_diag_data * sum^(-1) */
                  /* note that results are uneffected for most problems if
                     we do sum^(-1) * A_diag_data - but it seems to matter
                     a little for very non-sym */

                  if (hypre_CSRBlockMatrixBlockMultInv(sum_block, &A_diag_data[jj * bnnz],
                                                       distribute_block, block_size) == 0)
                  {


                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row )
                        {

                           /*  P_diag_data[P_marker[i2]]
                               += distribute * A_diag_data[jj1];*/

                           /* multiply - result in sum_block */
                           hypre_CSRBlockMatrixBlockMultAdd(distribute_block,
                                                            &A_diag_data[jj1 * bnnz], 0.0,
                                                            sum_block, block_size);


                           /* add result to p_diag_data */
                           hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                  &P_diag_data[P_marker[i2]*bnnz],
                                                                  block_size);

                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd)
                           {
                              /* P_offd_data[P_marker_offd[i2]]
                                 += distribute * A_offd_data[jj1]; */

                              /* multiply - result in sum_block */
                              hypre_CSRBlockMatrixBlockMultAdd(distribute_block,
                                                               &A_offd_data[jj1 * bnnz], 0.0,
                                                               sum_block, block_size);


                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                     &P_offd_data[P_marker_offd[i2]*bnnz],
                                                                     block_size);
                           }
                        }
                     }
                  }
                  else /* sum block is all zeros (or almost singular) - just add to diagonal */
                  {
                     /* diagonal += A_diag_data[jj]; */
                     if (add_weak_to_diag) hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                                                     diagonal_block,
                                                                                     block_size);

                  }
               }

               /*--------------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3 && add_weak_to_diag)
               {
                  /* diagonal += A_diag_data[jj];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         diagonal_block,
                                                         block_size);

               }

            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate( &A_offd_data[jj * bnnz],
                                                             &P_offd_data[P_marker_offd[i1]*bnnz],
                                                             block_size);
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and strongly influences i,
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker || (!add_weak_to_diag  && CF_marker[i1] != -3))
                  {

                     /* initialize sum to zero */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block,
                                                       1.0, block_size);

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.
                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulate(&A_ext_data[jj1 * bnnz],
                                                                     sum_block, block_size);
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulate(&A_ext_data[jj1 * bnnz],
                                                                     sum_block, block_size);

                           }
                        }
                     }

                     /* check whether sum_block is singular */


                     /* distribute = A_offd_data[jj] / sum;  */
                     /* here we want: A_offd_data * sum^(-1) */
                     if (hypre_CSRBlockMatrixBlockMultInv(sum_block, &A_offd_data[jj * bnnz],
                                                          distribute_block, block_size) == 0)
                     {

                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.
                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row)
                              {
                                 /* P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1]; */

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAdd(distribute_block,
                                                                  &A_ext_data[jj1 * bnnz], 0.0,
                                                                  sum_block, block_size);


                                 /* add result to p_diag_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_diag_data[P_marker[i2]*bnnz],
                                                                        block_size);

                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)

                                 /*P_offd_data[P_marker_offd[-i2-1]]
                                   += distribute * A_ext_data[jj1];*/
                              {

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAdd(distribute_block,
                                                                  &A_ext_data[jj1 * bnnz], 0.0,
                                                                  sum_block, block_size);


                                 /* add result to p_offd_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_offd_data[P_marker_offd[-i2 - 1]*bnnz],
                                                                        block_size);
                              }


                           }
                        }
                     }
                     else /* sum block is all zeros - just add to diagonal */
                     {
                        /* diagonal += A_offd_data[jj]; */
                        if (add_weak_to_diag) hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj * bnnz],
                                                                                        diagonal_block,
                                                                                        block_size);

                     }
                  }

                  /*-----------------------------------------------------------
                   * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                   * into the diagonal.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3 && add_weak_to_diag)
                  {
                     /* diagonal += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj * bnnz],
                                                            diagonal_block,
                                                            block_size);

                  }
               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {

               /* P_diag_data[jj] /= -diagonal; */

               /* want diagonal^(-1)*P_diag_data */
               /* do division - put in sum_block */
               if ( hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_diag_data[jj * bnnz],
                                                     sum_block, block_size) == 0)
               {
                  /* now copy to  P_diag_data[jj] and make negative */
                  hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
               else
               {
                  /* hypre_printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
                  /* just make P_diag_data negative since diagonal is singular) */
                  hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj * bnnz], &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);

               }
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               /* P_offd_data[jj] /= -diagonal; */

               /* do division - put in sum_block */
               hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_offd_data[jj * bnnz],
                                                sum_block, block_size);

               /* now copy to  P_offd_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj * bnnz],
                                                 -1.0, block_size);



            }

         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                     hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                     total_global_cpts,
                                     hypre_ParCSRBlockMatrixColStarts(A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine]);

   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0  || max_elmts > 0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   if (num_cols_P_offd)
   {
      hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P, A, tmp_map_offd, fine_to_coarse_offd);


   *P_ptr = P;


   hypre_TFree(zero_block, HYPRE_MEMORY_HOST);
   hypre_TFree(identity_block, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal_block, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute_block, HYPRE_MEMORY_HOST);

   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   if (num_procs > 1) { hypre_CSRBlockMatrixDestroy(A_ext); }

   return hypre_error_flag;
}

/* 8/07 - not sure that it is appropriate to scale by the blocks - for
   now it is commented out - may want to change this or do something
   different  */

HYPRE_Int
hypre_BoomerAMGBlockInterpTruncation( hypre_ParCSRBlockMatrix *P,
                                      HYPRE_Real    trunc_factor,
                                      HYPRE_Int max_elmts)
{
   hypre_CSRBlockMatrix *P_diag = hypre_ParCSRBlockMatrixDiag(P);
   HYPRE_Int     *P_diag_i = hypre_CSRBlockMatrixI(P_diag);
   HYPRE_Int     *P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
   HYPRE_Real    *P_diag_data = hypre_CSRBlockMatrixData(P_diag);
   HYPRE_Int     *P_diag_j_new;
   HYPRE_Real    *P_diag_data_new;

   hypre_CSRBlockMatrix *P_offd = hypre_ParCSRBlockMatrixOffd(P);
   HYPRE_Int     *P_offd_i = hypre_CSRBlockMatrixI(P_offd);
   HYPRE_Int     *P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
   HYPRE_Real    *P_offd_data = hypre_CSRBlockMatrixData(P_offd);
   HYPRE_Int     *P_offd_j_new;
   HYPRE_Real    *P_offd_data_new;

   HYPRE_Int   block_size = hypre_CSRBlockMatrixBlockSize(P_diag);
   HYPRE_Int   bnnz = block_size * block_size;

   HYPRE_Int   n_fine = hypre_CSRBlockMatrixNumRows(P_diag);
   HYPRE_Int   num_cols = hypre_CSRBlockMatrixNumCols(P_diag);
   HYPRE_Int   i, j, start_j, k;
   HYPRE_Int   ierr = 0;
   HYPRE_Int   next_open = 0;
   HYPRE_Int   now_checking = 0;
   HYPRE_Int   num_lost = 0;
   HYPRE_Int   next_open_offd = 0;
   HYPRE_Int   now_checking_offd = 0;
   HYPRE_Int   num_lost_offd = 0;
   HYPRE_Int   P_diag_size;
   HYPRE_Int   P_offd_size;
   HYPRE_Real  max_coef, tmp;
   HYPRE_Real *row_sum;
   HYPRE_Real *scale;
   HYPRE_Real *out_block;
   HYPRE_Int   cnt, cnt_diag, cnt_offd;
   HYPRE_Int   num_elmts;

   /* for now we will use the frobenius norm to
      determine whether to keep a block or not  - so norm_type = 1*/
   row_sum  = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   scale = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   out_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   if (trunc_factor > 0)
   {
      /* go through each row */
      for (i = 0; i < n_fine; i++)
      {
         max_coef = 0.0;

         /* diag */
         for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
         {
            hypre_CSRBlockMatrixBlockNorm(1, &P_diag_data[j * bnnz], &tmp, block_size);
            max_coef = (max_coef < tmp) ?  tmp : max_coef;
         }

         /* off_diag */
         for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
         {
            hypre_CSRBlockMatrixBlockNorm(1, &P_offd_data[j * bnnz], &tmp, block_size);
            max_coef = (max_coef < tmp) ?  tmp : max_coef;
         }

         max_coef *= trunc_factor;

         start_j = P_diag_i[i];
         P_diag_i[i] -= num_lost;

         /* set scale and row sum to zero */
         hypre_CSRBlockMatrixBlockSetScalar(scale, 0.0, block_size);
         hypre_CSRBlockMatrixBlockSetScalar(row_sum, 0.0, block_size);

         for (j = start_j; j < P_diag_i[i + 1]; j++)
         {
            /* row_sum += P_diag_data[now_checking];*/
            hypre_CSRBlockMatrixBlockAddAccumulate(&P_diag_data[now_checking * bnnz], row_sum, block_size);

            hypre_CSRBlockMatrixBlockNorm(1, &P_diag_data[now_checking * bnnz], &tmp, block_size);

            if ( tmp < max_coef)
            {
               num_lost++;
               now_checking++;
            }
            else
            {
               /* scale += P_diag_data[now_checking]; */
               hypre_CSRBlockMatrixBlockAddAccumulate(&P_diag_data[now_checking * bnnz], scale, block_size);

               /* P_diag_data[next_open] = P_diag_data[now_checking]; */
               hypre_CSRBlockMatrixBlockCopyData( &P_diag_data[now_checking * bnnz],
                                                  &P_diag_data[next_open * bnnz],
                                                  1.0, block_size);

               P_diag_j[next_open] = P_diag_j[now_checking];
               now_checking++;
               next_open++;
            }
         }

         start_j = P_offd_i[i];
         P_offd_i[i] -= num_lost_offd;

         for (j = start_j; j < P_offd_i[i + 1]; j++)
         {
            /* row_sum += P_offd_data[now_checking_offd]; */
            hypre_CSRBlockMatrixBlockAddAccumulate(&P_offd_data[now_checking_offd * bnnz], row_sum, block_size);

            hypre_CSRBlockMatrixBlockNorm(1, &P_offd_data[now_checking_offd * bnnz], &tmp, block_size);

            if ( tmp < max_coef)
            {
               num_lost_offd++;
               now_checking_offd++;
            }
            else
            {
               /* scale += P_offd_data[now_checking_offd]; */
               hypre_CSRBlockMatrixBlockAddAccumulate(&P_offd_data[now_checking_offd * bnnz], scale, block_size);

               /* P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];*/
               hypre_CSRBlockMatrixBlockCopyData( &P_offd_data[now_checking_offd * bnnz],
                                                  &P_offd_data[next_open_offd * bnnz],
                                                  1.0, block_size);


               P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
               now_checking_offd++;
               next_open_offd++;
            }
         }
         /* normalize row of P */
#if 0
         /* out_block = row_sum/scale; */
         if (hypre_CSRBlockMatrixBlockInvMult(scale, row_sum, out_block, block_size) == 0)
         {

            for (j = P_diag_i[i]; j < (P_diag_i[i + 1] - num_lost); j++)
            {
               /* P_diag_data[j] *= out_block; */

               /* put mult result in row_sum */
               hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_diag_data[j * bnnz], 0.0,
                                                row_sum, block_size);
               /* add to P_diag_data */
               hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_diag_data[j * bnnz], block_size);
            }

            for (j = P_offd_i[i]; j < (P_offd_i[i + 1] - num_lost_offd); j++)
            {

               /* P_offd_data[j] *= out_block; */

               /* put mult result in row_sum */
               hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_offd_data[j * bnnz], 0.0,
                                                row_sum, block_size);
               /* add to to P_offd_data */
               hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_offd_data[j * bnnz], block_size);

            }

         }
#endif
      }

      P_diag_i[n_fine] -= num_lost;
      P_offd_i[n_fine] -= num_lost_offd;
   }
   if (max_elmts > 0)
   {
      HYPRE_Int   P_mxnum, cnt1, rowlength;
      HYPRE_Int  *P_aux_j;
      HYPRE_Real *P_aux_data;
      HYPRE_Real *norm_array;

      rowlength = 0;
      if (n_fine)
      {
         rowlength = P_diag_i[1] + P_offd_i[1];
      }
      P_mxnum = rowlength;
      for (i = 1; i < n_fine; i++)
      {
         rowlength = P_diag_i[i + 1] - P_diag_i[i] + P_offd_i[i + 1] - P_offd_i[i];
         if (rowlength > P_mxnum) { P_mxnum = rowlength; }
      }
      if (P_mxnum > max_elmts)
      {
         P_aux_j = hypre_CTAlloc(HYPRE_Int,  P_mxnum, HYPRE_MEMORY_HOST);
         P_aux_data = hypre_CTAlloc(HYPRE_Real,  P_mxnum * bnnz, HYPRE_MEMORY_HOST);
         cnt_diag = 0;
         cnt_offd = 0;

         for (i = 0; i < n_fine; i++)
         {
            hypre_CSRBlockMatrixBlockSetScalar(row_sum, 0.0, block_size);
            /*row_sum = 0; */

            num_elmts = P_diag_i[i + 1] - P_diag_i[i] + P_offd_i[i + 1] - P_offd_i[i];
            if (max_elmts < num_elmts)
            {
               cnt = 0;
               for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
               {
                  P_aux_j[cnt] = P_diag_j[j];
                  /*P_aux_data[cnt++] = P_diag_data[j];*/
                  hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[j * bnnz],
                                                    &P_aux_data[cnt * bnnz],
                                                    1.0, block_size);
                  cnt++;
                  /*row_sum += P_diag_data[j];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&P_diag_data[j * bnnz], row_sum, block_size);


               }
               num_lost += cnt;
               cnt1 = cnt;
               for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
               {
                  P_aux_j[cnt] = P_offd_j[j] + num_cols;
                  /*P_aux_data[cnt++] = P_offd_data[j];*/
                  hypre_CSRBlockMatrixBlockCopyData(&P_offd_data[j * bnnz],
                                                    &P_aux_data[cnt * bnnz],
                                                    1.0, block_size);
                  cnt++;

                  /*row_sum += P_offd_data[j];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&P_offd_data[j * bnnz], row_sum, block_size);


               }
               num_lost_offd += cnt - cnt1;
               /* sort data */
               norm_array = hypre_CTAlloc(HYPRE_Real,  cnt, HYPRE_MEMORY_HOST);
               for (j = 0; j < cnt; j++)
               {
                  hypre_CSRBlockMatrixBlockNorm(1, &P_aux_data[j * bnnz], &norm_array[j], block_size);
               }

               hypre_block_qsort(P_aux_j, norm_array, P_aux_data, block_size, 0, cnt - 1);

               hypre_TFree(norm_array, HYPRE_MEMORY_HOST);

               /* scale = 0; */
               hypre_CSRBlockMatrixBlockSetScalar(scale, 0.0, block_size);
               P_diag_i[i] = cnt_diag;
               P_offd_i[i] = cnt_offd;
               for (j = 0; j < max_elmts; j++)
               {
                  /* scale += P_aux_data[j];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&P_aux_data[j * bnnz],
                                                         scale, block_size);


                  if (P_aux_j[j] < num_cols)
                  {
                     P_diag_j[cnt_diag] = P_aux_j[j];
                     /*P_diag_data[cnt_diag++] = P_aux_data[j];*/
                     hypre_CSRBlockMatrixBlockCopyData(&P_aux_data[j * bnnz],
                                                       &P_diag_data[cnt_diag * bnnz],
                                                       1.0, block_size);

                     cnt_diag++;


                  }
                  else
                  {
                     P_offd_j[cnt_offd] = P_aux_j[j] - num_cols;
                     /*P_offd_data[cnt_offd++] = P_aux_data[j];*/
                     hypre_CSRBlockMatrixBlockCopyData(&P_aux_data[j * bnnz],
                                                       &P_offd_data[cnt_offd * bnnz],
                                                       1.0, block_size);
                     cnt_offd++;

                  }
               }
               num_lost -= cnt_diag - P_diag_i[i];
               num_lost_offd -= cnt_offd - P_offd_i[i];
               /* normalize row of P */
               /* out_block = row_sum/scale; */
               /*if (scale != 0.)*/
#if 0
               if (hypre_CSRBlockMatrixBlockInvMult(scale, row_sum, out_block, block_size) == 0)
               {

                  for (j = P_diag_i[i]; j < cnt_diag; j++)
                  {

                     /* P_diag_data[j] *= out_block; */

                     /* put mult result in row_sum */
                     hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_diag_data[j * bnnz], 0.0,
                                                      row_sum, block_size);
                     /* add to P_diag_data */
                     hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_diag_data[j * bnnz], block_size);
                  }

                  for (j = P_offd_i[i]; j < cnt_offd; j++)
                  {

                     /* P_offd_data[j] *= out_block; */

                     /* put mult result in row_sum */
                     hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_offd_data[j * bnnz], 0.0,
                                                      row_sum, block_size);
                     /* add to to P_offd_data */
                     hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_offd_data[j * bnnz], block_size);
                  }


               }
#endif
            }
            else
            {
               if (P_diag_i[i] != cnt_diag)
               {
                  start_j = P_diag_i[i];
                  P_diag_i[i] = cnt_diag;
                  for (j = start_j; j < P_diag_i[i + 1]; j++)
                  {
                     P_diag_j[cnt_diag] = P_diag_j[j];
                     /*P_diag_data[cnt_diag++] = P_diag_data[j];*/
                     hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[j * bnnz],
                                                       &P_diag_data[cnt_diag * bnnz],
                                                       1.0, block_size);
                     cnt_diag++;


                  }
               }
               else
               {
                  cnt_diag += P_diag_i[i + 1] - P_diag_i[i];
               }
               if (P_offd_i[i] != cnt_offd)
               {
                  start_j = P_offd_i[i];
                  P_offd_i[i] = cnt_offd;
                  for (j = start_j; j < P_offd_i[i + 1]; j++)
                  {
                     P_offd_j[cnt_offd] = P_offd_j[j];
                     /*P_offd_data[cnt_offd++] = P_offd_data[j];*/

                     hypre_CSRBlockMatrixBlockCopyData(&P_offd_data[j * bnnz],
                                                       &P_offd_data[cnt_offd * bnnz],
                                                       1.0, block_size);
                     cnt_offd++;
                  }
               }
               else
               {
                  cnt_offd += P_offd_i[i + 1] - P_offd_i[i];
               }
            }
         }
         P_diag_i[n_fine] = cnt_diag;
         P_offd_i[n_fine] = cnt_offd;
         hypre_TFree(P_aux_j, HYPRE_MEMORY_HOST);
         hypre_TFree(P_aux_data, HYPRE_MEMORY_HOST);
      }
   }




   if (num_lost)
   {
      P_diag_size = P_diag_i[n_fine];
      P_diag_j_new = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
      P_diag_data_new = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);
      for (i = 0; i < P_diag_size; i++)
      {
         P_diag_j_new[i] = P_diag_j[i];
         for (k = 0; k < bnnz; k++)
         {
            P_diag_data_new[i * bnnz + k] = P_diag_data[i * bnnz + k];
         }

      }
      hypre_TFree(P_diag_j, HYPRE_MEMORY_HOST);
      hypre_TFree(P_diag_data, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_size;
   }
   if (num_lost_offd)
   {
      P_offd_size = P_offd_i[n_fine];
      P_offd_j_new = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
      P_offd_data_new = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);
      for (i = 0; i < P_offd_size; i++)
      {
         P_offd_j_new[i] = P_offd_j[i];
         for (k = 0; k < bnnz; k++)
         {
            P_offd_data_new[i * bnnz + k] = P_offd_data[i * bnnz + k];
         }

      }
      hypre_TFree(P_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(P_offd_data, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixJ(P_offd) = P_offd_j_new;
      hypre_CSRMatrixData(P_offd) = P_offd_data_new;
      hypre_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
   }

   hypre_TFree(row_sum, HYPRE_MEMORY_HOST);
   hypre_TFree(scale, HYPRE_MEMORY_HOST);
   hypre_TFree(out_block, HYPRE_MEMORY_HOST);

   return ierr;
}

/*-----------------------------------------------*/
/* compare on w, move v and blk_array */

void hypre_block_qsort( HYPRE_Int  *v,
                        HYPRE_Complex *w,
                        HYPRE_Complex *blk_array,
                        HYPRE_Int   block_size,
                        HYPRE_Int   left,
                        HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }

   hypre_swap2( v, w, left, (left + right) / 2);
   hypre_swap_blk(blk_array, block_size, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
      if (hypre_cabs(w[i]) > hypre_cabs(w[left]))
      {
         hypre_swap2(v, w, ++last, i);
         hypre_swap_blk(blk_array, block_size, last, i);
      }
   hypre_swap2(v, w, left, last);
   hypre_swap_blk(blk_array, block_size, left, last);
   hypre_block_qsort(v, w, blk_array, block_size, left, last - 1);
   hypre_block_qsort(v, w, blk_array, block_size, last + 1, right);
}

void hypre_swap_blk( HYPRE_Complex *v,
                     HYPRE_Int   block_size,
                     HYPRE_Int   i,
                     HYPRE_Int   j )
{
   HYPRE_Int bnnz = block_size * block_size;
   HYPRE_Real    *temp;

   temp = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   /*temp = v[i];*/
   hypre_CSRBlockMatrixBlockCopyData(&v[i * bnnz], temp, 1.0, block_size);
   /*v[i] = v[j];*/
   hypre_CSRBlockMatrixBlockCopyData(&v[j * bnnz], &v[i * bnnz], 1.0, block_size);
   /* v[j] = temp; */
   hypre_CSRBlockMatrixBlockCopyData(temp, &v[j * bnnz], 1.0, block_size);

   hypre_TFree(temp, HYPRE_MEMORY_HOST);
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterpDiag

 This is the block version of classical R-S interpolation. We use just the
 diagonals of these blocks.

 A and P are now Block matrices.  The Strength matrix S is not as it gives
 nodal strengths.

 CF_marker is size number of nodes.

 add_weak_to_diag  0 = don't add weak connections to diag (distribute instead)
 1 = do add
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockInterpDiag( hypre_ParCSRBlockMatrix  *A,
                                     HYPRE_Int                *CF_marker,
                                     hypre_ParCSRMatrix       *S,
                                     HYPRE_BigInt             *num_cpts_global,
                                     HYPRE_Int                 num_functions,
                                     HYPRE_Int                *dof_func,
                                     HYPRE_Int                 debug_flag,
                                     HYPRE_Real                trunc_factor,
                                     HYPRE_Int                 max_elmts,
                                     HYPRE_Int                 add_weak_to_diag,
                                     hypre_ParCSRBlockMatrix **P_ptr)
{
   HYPRE_UNUSED_VAR(dof_func);

   MPI_Comm                 comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix  *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real            *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   HYPRE_Int              block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int              bnnz = block_size * block_size;

   hypre_CSRBlockMatrix  *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real            *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_BigInt          *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix       *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix       *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   HYPRE_BigInt          *col_map_offd_P;
   HYPRE_Int             *tmp_map_offd = NULL;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRBlockMatrix  *A_ext = NULL;
   HYPRE_Real            *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_BigInt          *A_ext_j = NULL;

   hypre_CSRBlockMatrix  *P_diag;
   hypre_CSRBlockMatrix  *P_offd;

   HYPRE_Real            *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Real            *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;

   HYPRE_Int             *P_marker, *P_marker_offd = NULL;

   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd = NULL;
   HYPRE_Int              jj_begin_row, jj_begin_row_offd;
   HYPRE_Int              jj_end_row, jj_end_row_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_BigInt          *fine_to_coarse_offd = NULL;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_BigInt           my_first_cpt, total_global_cpts;
   HYPRE_Int              num_cols_P_offd;

   HYPRE_Int              bd;

   HYPRE_Int              i, i1, i2;
   HYPRE_Int              j, jl, jj, jj1;
   HYPRE_Int              kc;
   HYPRE_BigInt           big_k;
   HYPRE_Int              start;

   HYPRE_Int              c_num;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data = NULL;
   HYPRE_BigInt          *big_buf_data = NULL;

   HYPRE_BigInt col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   HYPRE_Int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_BigInt col_n = col_1 + (HYPRE_BigInt)local_numrows;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */


   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block;
   HYPRE_Real       *distribute_block;

   HYPRE_Real       *sign;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   if (num_functions > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for num_functions > 1!");
   }

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

   }

   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);


   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixBigJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            /* for the data field we must get all of the block data */
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (HYPRE_BigInt)(-kc - 1);
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
               }
               index++;
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/


   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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


      /* loop over the fine grid points */
      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector (fine_to_coarse is the mapping vector).
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real, bnnz, HYPRE_MEMORY_HOST);
   zero_block     = hypre_CTAlloc(HYPRE_Real, bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }

   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block   = hypre_CTAlloc(HYPRE_Real, bnnz, HYPRE_MEMORY_HOST);
   sum_block        = hypre_CTAlloc(HYPRE_Real, bnnz, HYPRE_MEMORY_HOST);
   distribute_block = hypre_CTAlloc(HYPRE_Real, bnnz, HYPRE_MEMORY_HOST);

   sign = hypre_CTAlloc(HYPRE_Real, block_size, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         big_buf_data[index++] = my_first_cpt
                                 + fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */

   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data, fine_to_coarse_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }

      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      strong_f_marker = -2;

      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            /* P_diag_data[jj_counter] = one; */
            hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                              &P_diag_data[jj_counter * bnnz],
                                              1.0, block_size);
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     /* P_offd_data[jj_counter_offd] = zero; */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);

                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;


            /* get the diagonal block */
            /* diagonal = A_diag_data[A_diag_i[i]]; */
            hypre_CSRBlockMatrixBlockCopyDataDiag(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block,
                                                  1.0, block_size);



            /* Here we go through the neighborhood of this grid point */

            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj * bnnz],
                                                             &P_diag_data[P_marker[i1]*bnnz],
                                                             block_size);

               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker || (!add_weak_to_diag  && CF_marker[i1] != -3))
               {
                  /* initialize sum to zero */
                  /* sum = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0,
                                                    block_size);


                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *-----------------------------------------------------------*/

                  hypre_CSRBlockMatrixComputeSign(&A_diag_data[A_diag_i[i1]*bnnz], sign, block_size);


                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row)
                     {
                        /* add diag data to sum */
                        /* sum += A_diag_data[jj1]; */
                        /* hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj1*bnnz],
                           sum_block, block_size);*/

                        hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(&A_diag_data[jj1 * bnnz],
                                                                            sum_block, block_size, sign);
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd )
                        {
                           /* add off diag data to sum */
                           /*sum += A_offd_data[jj1];*/
                           /* hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj1*bnnz],
                              sum_block, block_size);*/
                           hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(&A_offd_data[jj1 * bnnz],
                                                                               sum_block, block_size, sign);
                        }
                     }
                  }
                  /* check whether sum_block is singular */

                  /* distribute = A_diag_data[jj] / sum;*/
                  /* here we want: A_diag_data * sum^(-1) */

                  if (hypre_CSRBlockMatrixBlockInvMultDiag(sum_block, &A_diag_data[jj * bnnz],
                                                           distribute_block, block_size) == 0)
                  {


                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row )
                        {

                           /*  P_diag_data[P_marker[i2]]
                               += distribute * A_diag_data[jj1];*/

                           /* multiply - result in sum_block */
                           hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                             sum_block, 1.0, block_size);


                           /* hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block,
                              &A_diag_data[jj1*bnnz], 0.0,
                              sum_block, block_size);*/
                           hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(distribute_block,
                                                                         &A_diag_data[jj1 * bnnz], 0.0,
                                                                         sum_block, block_size, sign);


                           /* add result to p_diag_data */
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block,
                                                                      &P_diag_data[P_marker[i2]*bnnz],
                                                                      block_size);

                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd)
                           {
                              /* P_offd_data[P_marker_offd[i2]]
                                 += distribute * A_offd_data[jj1]; */

                              /* multiply - result in sum_block */

                              hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                                sum_block, 1.0, block_size);
                              /* hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block,
                                 &A_offd_data[jj1*bnnz], 0.0,
                                 sum_block, block_size); */

                              hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(distribute_block,
                                                                            &A_offd_data[jj1 * bnnz], 0.0,
                                                                            sum_block, block_size, sign);



                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block,
                                                                         &P_offd_data[P_marker_offd[i2]*bnnz],
                                                                         block_size);


                           }
                        }
                     }
                  }
                  else /* sum block is all zeros (or almost singular) - just add to diagonal */
                  {
                     /* diagonal += A_diag_data[jj]; */
                     if (add_weak_to_diag) hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj * bnnz],
                                                                                         diagonal_block,
                                                                                         block_size);
                  }
               }

               /*--------------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3 && add_weak_to_diag)
               {
                  /* diagonal += A_diag_data[jj];*/
                  hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj * bnnz],
                                                             diagonal_block,
                                                             block_size);
               }
            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulateDiag( &A_offd_data[jj * bnnz],
                                                                 &P_offd_data[P_marker_offd[i1]*bnnz],
                                                                 block_size);
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and strongly influences i,
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker || (!add_weak_to_diag  && CF_marker[i1] != -3))
                  {

                     /* initialize sum to zero */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block,
                                                       1.0, block_size);

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.
                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     hypre_CSRBlockMatrixComputeSign(&A_ext_data[A_ext_i[c_num]*bnnz], sign, block_size);


                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* sum += A_ext_data[jj1]; */
                              /*  hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1*bnnz],
                                  sum_block, block_size);*/
                              hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(&A_ext_data[jj1 * bnnz],
                                                                                  sum_block, block_size, sign);
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)
                           {
                              /* sum += A_ext_data[jj1]; */
                              /* hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1*bnnz],
                                 sum_block, block_size);*/
                              hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(&A_ext_data[jj1 * bnnz],
                                                                                  sum_block, block_size, sign);
                           }
                        }
                     }

                     /* check whether sum_block is singular */


                     /* distribute = A_offd_data[jj] / sum;  */
                     /* here we want: A_offd_data * sum^(-1) */
                     if (hypre_CSRBlockMatrixBlockInvMultDiag(sum_block, &A_offd_data[jj * bnnz],
                                                              distribute_block, block_size) == 0)
                     {

                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.
                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row)
                              {
                                 /* P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1]; */

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, block_size);

                                 /* hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block,
                                    &A_ext_data[jj1*bnnz], 0.0,
                                    sum_block, block_size); */

                                 hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(distribute_block,
                                                                               &A_ext_data[jj1 * bnnz], 0.0,
                                                                               sum_block, block_size, sign);
                                 /* add result to p_diag_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block,
                                                                            &P_diag_data[P_marker[i2]*bnnz],
                                                                            block_size);
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)

                                 /*P_offd_data[P_marker_offd[-i2-1]]
                                   += distribute * A_ext_data[jj1];*/
                              {

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, block_size);

                                 /* hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block,
                                    &A_ext_data[jj1*bnnz], 0.0,
                                    sum_block, block_size);*/

                                 hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(distribute_block,
                                                                               &A_ext_data[jj1 * bnnz], 0.0,
                                                                               sum_block, block_size, sign);
                                 /* add result to p_offd_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block,
                                                                            &P_offd_data[P_marker_offd[-i2 - 1]*bnnz],
                                                                            block_size);
                              }
                           }
                        }
                     }
                     else /* sum block is all zeros - just add to diagonal */
                     {
                        /* diagonal += A_offd_data[jj]; */
                        if (add_weak_to_diag) hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj * bnnz],
                                                                                            diagonal_block,
                                                                                            block_size);
                     }
                  }

                  /*-----------------------------------------------------------
                   * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                   * into the diagonal.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3 && add_weak_to_diag)
                  {
                     /* diagonal += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj * bnnz],
                                                                diagonal_block, block_size);
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {

               /* P_diag_data[jj] /= -diagonal; */

               /* want diagonal^(-1)*P_diag_data */
               /* do division - put in sum_block */
               if ( hypre_CSRBlockMatrixBlockInvMultDiag(diagonal_block, &P_diag_data[jj * bnnz],
                                                         sum_block, block_size) == 0)
               {
                  /* now copy to  P_diag_data[jj] and make negative */
                  hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
               else
               {
                  /* hypre_printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
                  /* just make P_diag_data negative since diagonal is zero */
                  hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj * bnnz], &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               /* P_offd_data[jj] /= -diagonal; */

               /* do division - put in sum_block */
               hypre_CSRBlockMatrixBlockInvMultDiag(diagonal_block, &P_offd_data[jj * bnnz],
                                                    sum_block, block_size);

               /* now copy to  P_offd_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj * bnnz],
                                                 -1.0, block_size);
            }
         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                     hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                     total_global_cpts,
                                     hypre_ParCSRBlockMatrixColStarts(A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine]);


   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }


   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   if (num_cols_P_offd)
   {
      hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P, A, tmp_map_offd, fine_to_coarse_offd);


   *P_ptr = P;

   hypre_TFree(sign, HYPRE_MEMORY_HOST);


   hypre_TFree(zero_block, HYPRE_MEMORY_HOST);
   hypre_TFree(identity_block, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal_block, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute_block, HYPRE_MEMORY_HOST);

   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   if (num_procs > 1) { hypre_CSRBlockMatrixDestroy(A_ext); }

   return (0);

}


/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterpRV

 Here we are modifying the block interp like in Ruge's elasticity paper
 (applied math comp '86) - only we don't include the diagonal
 for dist. the f-connect


 - when we do the distribution of the f-connection, we only distribute the error
 to like unknowns - this has the effect of only using the diagonal of the
 matrix for the f-distributions.  In addition, we will not differentiate
 between the strength of the f-connections (so nothing is added to the diag)

 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV( hypre_ParCSRBlockMatrix    *A,
                                   HYPRE_Int                  *CF_marker,
                                   hypre_ParCSRMatrix         *S,
                                   HYPRE_BigInt               *num_cpts_global,
                                   HYPRE_Int                   num_functions,
                                   HYPRE_Int                  *dof_func,
                                   HYPRE_Int                   debug_flag,
                                   HYPRE_Real                  trunc_factor,
                                   HYPRE_Int                   max_elmts,
                                   hypre_ParCSRBlockMatrix   **P_ptr)
{
   HYPRE_UNUSED_VAR(dof_func);

   MPI_Comm                 comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix  *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real            *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   HYPRE_Int              block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int              bnnz = block_size * block_size;

   hypre_CSRBlockMatrix  *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real            *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_BigInt          *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix       *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix       *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   HYPRE_BigInt            *col_map_offd_P;
   HYPRE_Int               *tmp_map_offd = NULL;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRBlockMatrix  *A_ext = NULL;
   HYPRE_Real            *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_BigInt          *A_ext_j = NULL;

   hypre_CSRBlockMatrix  *P_diag;
   hypre_CSRBlockMatrix  *P_offd;

   HYPRE_Real            *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Real            *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;

   HYPRE_Int             *P_marker, *P_marker_offd = NULL;

   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd = NULL;
   HYPRE_Int              jj_begin_row, jj_begin_row_offd;
   HYPRE_Int              jj_end_row, jj_end_row_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_BigInt          *fine_to_coarse_offd = NULL;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_BigInt           total_global_cpts;
   HYPRE_Int              num_cols_P_offd;
   HYPRE_BigInt           my_first_cpt;

   HYPRE_Int              bd;

   HYPRE_Int              i, i1, i2;
   HYPRE_Int              j, jl, jj, jj1;
   HYPRE_Int              kc;
   HYPRE_BigInt           big_k;
   HYPRE_Int              start;

   HYPRE_Int              c_num;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data = NULL;
   HYPRE_BigInt          *big_buf_data = NULL;

   HYPRE_BigInt col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   HYPRE_Int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_BigInt col_n = col_1 + (HYPRE_BigInt)local_numrows;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */


   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block;
   HYPRE_Real       *distribute_block;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   if (num_functions > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for num_functions > 1!");
   }

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);


   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);


   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixBigJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            /* for the data field we must get all of the blocbig_k data */
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (HYPRE_BigInt)(-kc - 1);
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
               }
               index++;
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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


      /* loop over the fine grid points */
      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector (fine_to_coarse is the mapping vector).
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   zero_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }


   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   distribute_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);

   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         big_buf_data[index++] = my_first_cpt
                                 + fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */

   comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data,
                                               fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] -= my_first_cpt; }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }

      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      strong_f_marker = -2;

      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            /* P_diag_data[jj_counter] = one; */
            hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                              &P_diag_data[jj_counter * bnnz],
                                              1.0, block_size);
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     /* P_offd_data[jj_counter_offd] = zero; */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);

                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;


            /* get the diagonal block */
            /* diagonal = A_diag_data[A_diag_i[i]]; */
            hypre_CSRBlockMatrixBlockCopyData(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block,
                                              1.0, block_size);



            /* Here we go through the neighborhood of this grid point */

            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         &P_diag_data[P_marker[i1]*bnnz],
                                                         block_size);

               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point (MAY or MAY NOT strongly influences i),
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker || CF_marker[i1] != -3)
               {
                  /* initialize sum to zero */
                  /* sum = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0,
                                                    block_size);


                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.-

                   HERE WE ONLY WANT THE DIAG CONTIRBUTIONS (intra-unknown)

                   *-----------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row)
                     {
                        /* add diag data to sum */
                        /* sum += A_diag_data[jj1]; */
                        hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj1 * bnnz],
                                                                   sum_block, block_size);
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd )
                        {
                           /* add off diag data to sum */
                           /*sum += A_offd_data[jj1];*/
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj1 * bnnz],
                                                                      sum_block, block_size);

                        }
                     }
                  }
                  /* check whether sum_block is singular (NOW SUM IS A DIAG MATRIX)*/
                  /* distribute = A_diag_data[jj] / sum;  (if a diag element is 0 then
                     that col is scaled by 1 instead of 1/diag) - doesn'treturn 0*/
                  if (hypre_CSRBlockMatrixBlockInvMultDiag2(&A_diag_data[jj * bnnz], sum_block,
                                                            distribute_block, block_size) == 0)
                  {

                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.-
                      HERE AGAIN WE ONLY WANT TO DIST W/IN A LIKE UNKNOWN

                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row )
                        {

                           /*  P_diag_data[P_marker[i2]]
                               += distribute * A_diag_data[jj1];*/

                           /* multiply - result in sum_block */
                           hypre_CSRBlockMatrixBlockMultAddDiag2(distribute_block,
                                                                 &A_diag_data[jj1 * bnnz], 0.0,
                                                                 sum_block, block_size);


                           /* add result to p_diag_data */
                           hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                  &P_diag_data[P_marker[i2]*bnnz],
                                                                  block_size);

                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd)
                           {
                              /* P_offd_data[P_marker_offd[i2]]
                                 += distribute * A_offd_data[jj1]; */

                              /* multiply - result in sum_block */
                              hypre_CSRBlockMatrixBlockMultAddDiag2(distribute_block,
                                                                    &A_offd_data[jj1 * bnnz], 0.0,
                                                                    sum_block, block_size);


                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                     &P_offd_data[P_marker_offd[i2]*bnnz],
                                                                     block_size);
                           }
                        }
                     }
                  } /* end of if sum */
               }/* end of case 1 or case 2*/

            }/* end of loop of diag part */


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate( &A_offd_data[jj * bnnz],
                                                             &P_offd_data[P_marker_offd[i1]*bnnz],
                                                             block_size);
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and (MAY or MAY NOT strongly influences i),
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker || CF_marker[i1] != -3 )
                  {

                     /* initialize sum to zero */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block,
                                                       1.0, block_size);

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.


                      HERE WE ONLY WANT THE DIAG CONTIRBUTIONS (intra-unknown)

                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1 * bnnz],
                                                                         sum_block, block_size);
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1 * bnnz],
                                                                         sum_block, block_size);

                           }
                        }
                     }

                     /* check whether sum_block is singular */


                     /* distribute = A_offd_data[jj] / sum;  */
                     /* here we want: A_offd_data * sum^(-1) */
                     if (hypre_CSRBlockMatrixBlockInvMultDiag2(&A_offd_data[jj * bnnz], sum_block,
                                                               distribute_block, block_size) == 0)
                     {

                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.

                         HERE AGAIN WE ONLY WANT TO DIST W/IN A LIKE UNKNOWN

                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row)
                              {
                                 /* P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1]; */

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAddDiag2(distribute_block,
                                                                       &A_ext_data[jj1 * bnnz], 0.0,
                                                                       sum_block, block_size);


                                 /* add result to p_diag_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_diag_data[P_marker[i2]*bnnz],
                                                                        block_size);

                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)

                                 /*P_offd_data[P_marker_offd[-i2-1]]
                                   += distribute * A_ext_data[jj1];*/
                              {

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAddDiag2(distribute_block,
                                                                       &A_ext_data[jj1 * bnnz], 0.0,
                                                                       sum_block, block_size);


                                 /* add result to p_offd_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_offd_data[P_marker_offd[-i2 - 1]*bnnz],
                                                                        block_size);
                              }
                           }
                        }
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {

               /* P_diag_data[jj] /= -diagonal; */

               /* want diagonal^(-1)*P_diag_data */
               /* do division - put in sum_block */
               if ( hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_diag_data[jj * bnnz],
                                                     sum_block, block_size) == 0)
               {
                  /* now copy to  P_diag_data[jj] and make negative */
                  hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
               else
               {
                  /* hypre_printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
                  /* just make P_diag_data negative since diagonal is singular) */
                  hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj * bnnz], &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);

               }
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               /* P_offd_data[jj] /= -diagonal; */

               /* do division - put in sum_block */
               hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_offd_data[jj * bnnz],
                                                sum_block, block_size);

               /* now copy to  P_offd_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj * bnnz],
                                                 -1.0, block_size);



            }

         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                     hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                     total_global_cpts,
                                     hypre_ParCSRBlockMatrixColStarts(A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine]);

   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }


   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);


      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   if (num_cols_P_offd)
   {
      hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P, A, tmp_map_offd, fine_to_coarse_offd);


   *P_ptr = P;


   hypre_TFree(zero_block, HYPRE_MEMORY_HOST);
   hypre_TFree(identity_block, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal_block, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute_block, HYPRE_MEMORY_HOST);

   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   if (num_procs > 1) { hypre_CSRBlockMatrixDestroy(A_ext); }

   return (0);

}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterpRV2

 Here we are modifying the block interp like in Ruge's elasticity paper as above
 (applied math comp '86), only instead of using just the diagonals of the
 scaling matrices (for the fine connections), we use a diagonal matrix
 whose diag entries are the row sumes (like suggested in Tanya Clees thesis
 for direct interp)

 -again there is no differentiation for weak/strong f-connections

 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV2( hypre_ParCSRBlockMatrix   *A,
                                    HYPRE_Int                 *CF_marker,
                                    hypre_ParCSRMatrix        *S,
                                    HYPRE_BigInt              *num_cpts_global,
                                    HYPRE_Int                  num_functions,
                                    HYPRE_Int                 *dof_func,
                                    HYPRE_Int                  debug_flag,
                                    HYPRE_Real                 trunc_factor,
                                    HYPRE_Int                  max_elmts,
                                    hypre_ParCSRBlockMatrix  **P_ptr)
{
   HYPRE_UNUSED_VAR(dof_func);
   HYPRE_UNUSED_VAR(num_functions);

   MPI_Comm           comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real           *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int            *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int            *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   HYPRE_Int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int             bnnz = block_size * block_size;

   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_BigInt          *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   HYPRE_BigInt          *col_map_offd_P;
   HYPRE_Int             *tmp_map_offd = NULL;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRBlockMatrix  *A_ext = NULL;
   HYPRE_Real            *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_BigInt          *A_ext_j = NULL;

   hypre_CSRBlockMatrix    *P_diag;
   hypre_CSRBlockMatrix    *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i;
   HYPRE_Int       *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int       *P_offd_i;
   HYPRE_Int       *P_offd_j;

   HYPRE_Int        P_diag_size, P_offd_size;

   HYPRE_Int       *P_marker, *P_marker_offd = NULL;

   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int       *jj_count, *jj_count_offd = NULL;
   HYPRE_Int        jj_begin_row, jj_begin_row_offd;
   HYPRE_Int        jj_end_row, jj_end_row_offd;

   HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int        n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   HYPRE_Int        strong_f_marker;

   HYPRE_Int       *fine_to_coarse;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;
   HYPRE_Int       *coarse_counter;
   HYPRE_Int        coarse_shift;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_Int        num_cols_P_offd;
   HYPRE_BigInt     my_first_cpt;

   HYPRE_Int        bd;

   HYPRE_Int        i, i1, i2;
   HYPRE_Int        j, jl, jj, jj1;
   HYPRE_Int        kc;
   HYPRE_BigInt     big_k;
   HYPRE_Int        start;

   HYPRE_Int        c_num;

   HYPRE_Int        my_id;
   HYPRE_Int        num_procs;
   HYPRE_Int        num_threads;
   HYPRE_Int        num_sends;
   HYPRE_Int        index;
   HYPRE_Int        ns, ne, size, rest;
   HYPRE_Int       *int_buf_data = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;

   HYPRE_BigInt col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   HYPRE_Int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_BigInt col_n = col_1 + (HYPRE_BigInt)local_numrows;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */


   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block;
   HYPRE_Real       *distribute_block;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);


   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

   }

   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);


   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixBigJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            /* for the data field we must get all of the blocbig_k data */
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (HYPRE_BigInt)(-kc - 1);
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index * bnnz + bd] = A_ext_data[j * bnnz + bd];
               }
               index++;
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/


   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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


      /* loop over the fine grid points */
      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector (fine_to_coarse is the mapping vector).
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   zero_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }


   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   distribute_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);

   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         big_buf_data[index++] = my_first_cpt
                                 + fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */

   comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data,
                                               fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }


   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/


   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }

      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      strong_f_marker = -2;

      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            /* P_diag_data[jj_counter] = one; */
            hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                              &P_diag_data[jj_counter * bnnz],
                                              1.0, block_size);
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     /* P_offd_data[jj_counter_offd] = zero; */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);

                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;


            /* get the diagonal block */
            /* diagonal = A_diag_data[A_diag_i[i]]; */
            hypre_CSRBlockMatrixBlockCopyData(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block,
                                              1.0, block_size);



            /* Here we go through the neighborhood of this grid point */

            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         &P_diag_data[P_marker[i1]*bnnz],
                                                         block_size);

               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point (MAY or MAY NOT strongly influences i),
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker || CF_marker[i1] != -3)
               {
                  /* initialize sum to zero */
                  /* sum = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0,
                                                    block_size);

                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.-

                   HERE WE ONLY WANT THE DIAG CONTIRBUTIONS (intra-unknown)

                   *-----------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row)
                     {
                        /* add diag data to sum */
                        /* sum += A_diag_data[jj1]; */
                        hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj1 * bnnz],
                                                                   sum_block, block_size);
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd )
                        {
                           /* add off diag data to sum */
                           /*sum += A_offd_data[jj1];*/
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj1 * bnnz],
                                                                      sum_block, block_size);
                        }
                     }
                  }
                  /* check whether sum_block is singular (NOW SUM IS A DIAG MATRIX WHOSE
                     ENTRIES ARE THE ROW SUMS)*/
                  /* distribute = A_diag_data[jj] / sum;  (if a diag element is 0 then
                     that col is scaled by 1 instead of 1/diag) - doesn'treturn 0*/
                  if (hypre_CSRBlockMatrixBlockInvMultDiag3(&A_diag_data[jj * bnnz], sum_block,
                                                            distribute_block, block_size) == 0)
                  {

                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.-
                      (here we we use row-sums for the nodes recv. the distribution)
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row )
                        {

                           /*  P_diag_data[P_marker[i2]]
                               += distribute * A_diag_data[jj1];*/

                           /* multiply - result in sum_block */
                           hypre_CSRBlockMatrixBlockMultAddDiag3(distribute_block,
                                                                 &A_diag_data[jj1 * bnnz], 0.0,
                                                                 sum_block, block_size);

                           /* add result to p_diag_data */
                           hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                  &P_diag_data[P_marker[i2]*bnnz],
                                                                  block_size);
                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd)
                           {
                              /* P_offd_data[P_marker_offd[i2]]
                                 += distribute * A_offd_data[jj1]; */

                              /* multiply - result in sum_block */
                              hypre_CSRBlockMatrixBlockMultAddDiag3(distribute_block,
                                                                    &A_offd_data[jj1 * bnnz], 0.0,
                                                                    sum_block, block_size);


                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                     &P_offd_data[P_marker_offd[i2]*bnnz],
                                                                     block_size);
                           }
                        }
                     }
                  }
               }
            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate( &A_offd_data[jj * bnnz],
                                                             &P_offd_data[P_marker_offd[i1]*bnnz],
                                                             block_size);
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and (MAY or MAY NOT strongly influences i),
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker || CF_marker[i1] != -3 )
                  {

                     /* initialize sum to zero */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block,
                                                       1.0, block_size);

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.


                      HERE WE ONLY WANT THE DIAG CONTIRBUTIONS (intra-unknown)

                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1 * bnnz],
                                                                         sum_block, block_size);
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)
                           {
                              /* sum += A_ext_data[jj1]; */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1 * bnnz],
                                                                         sum_block, block_size);

                           }
                        }
                     }

                     /* check whether sum_block is singular */


                     /* distribute = A_offd_data[jj] / sum;  */
                     /* here we want: A_offd_data * sum^(-1)  - use the row sums as the
                        diag for sum*/
                     if (hypre_CSRBlockMatrixBlockInvMultDiag3(&A_offd_data[jj * bnnz], sum_block,
                                                               distribute_block, block_size) == 0)
                     {

                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.


                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row)
                              {
                                 /* P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1]; */

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAddDiag3(distribute_block,
                                                                       &A_ext_data[jj1 * bnnz], 0.0,
                                                                       sum_block, block_size);


                                 /* add result to p_diag_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_diag_data[P_marker[i2]*bnnz],
                                                                        block_size);
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd)

                                 /*P_offd_data[P_marker_offd[-i2-1]]
                                   += distribute * A_ext_data[jj1];*/
                              {

                                 /* multiply - result in sum_block */
                                 hypre_CSRBlockMatrixBlockMultAddDiag3(distribute_block,
                                                                       &A_ext_data[jj1 * bnnz], 0.0,
                                                                       sum_block, block_size);

                                 /* add result to p_offd_data */
                                 hypre_CSRBlockMatrixBlockAddAccumulate(sum_block,
                                                                        &P_offd_data[P_marker_offd[-i2 - 1]*bnnz],
                                                                        block_size);
                              }
                           }
                        }
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {

               /* P_diag_data[jj] /= -diagonal; */

               /* want diagonal^(-1)*P_diag_data */
               /* do division - put in sum_block */
               if ( hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_diag_data[jj * bnnz],
                                                     sum_block, block_size) == 0)
               {
                  /* now copy to  P_diag_data[jj] and make negative */
                  hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
               else
               {
                  /* hypre_printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
                  /* just make P_diag_data negative since diagonal is singular) */
                  hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj * bnnz], &P_diag_data[jj * bnnz],
                                                    -1.0, block_size);
               }
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               /* P_offd_data[jj] /= -diagonal; */

               /* do division - put in sum_block */
               hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_offd_data[jj * bnnz],
                                                sum_block, block_size);

               /* now copy to  P_offd_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj * bnnz],
                                                 -1.0, block_size);
            }
         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                     hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                     total_global_cpts,
                                     hypre_ParCSRBlockMatrixColStarts(A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine]);


   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   if (num_cols_P_offd)
   {
      hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P, A, tmp_map_offd, fine_to_coarse_offd);


   *P_ptr = P;


   hypre_TFree(zero_block, HYPRE_MEMORY_HOST);
   hypre_TFree(identity_block, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal_block, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute_block, HYPRE_MEMORY_HOST);

   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   if (num_procs > 1) { hypre_CSRBlockMatrixDestroy(A_ext); }

   return (0);

}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildBlockDirInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockDirInterp( hypre_ParCSRBlockMatrix    *A,
                                    HYPRE_Int                  *CF_marker,
                                    hypre_ParCSRMatrix         *S,
                                    HYPRE_BigInt               *num_cpts_global,
                                    HYPRE_Int                   num_functions,
                                    HYPRE_Int                  *dof_func,
                                    HYPRE_Int                   debug_flag,
                                    HYPRE_Real                  trunc_factor,
                                    HYPRE_Int                   max_elmts,
                                    hypre_ParCSRBlockMatrix   **P_ptr)
{
   HYPRE_UNUSED_VAR(dof_func);

   MPI_Comm           comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int            *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int            *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   HYPRE_Int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int             bnnz = block_size * block_size;


   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int            *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int            *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   HYPRE_Int             num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int            *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int            *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int            *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int            *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   HYPRE_BigInt         *col_map_offd_P;
   HYPRE_Int            *tmp_map_offd = NULL;

   HYPRE_Int            *CF_marker_offd = NULL;
   HYPRE_Int            *dof_func_offd = NULL;

   hypre_CSRBlockMatrix *P_diag;
   hypre_CSRBlockMatrix *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int            *P_diag_i;
   HYPRE_Int            *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int            *P_offd_i;
   HYPRE_Int            *P_offd_j;

   HYPRE_Int             P_diag_size, P_offd_size;

   HYPRE_Int            *P_marker, *P_marker_offd = NULL;

   HYPRE_Int             jj_counter, jj_counter_offd;
   HYPRE_Int            *jj_count, *jj_count_offd = NULL;
   HYPRE_Int             jj_begin_row, jj_begin_row_offd;
   HYPRE_Int             jj_end_row, jj_end_row_offd;

   HYPRE_Int             start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int             n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   HYPRE_Int            *fine_to_coarse;
   HYPRE_BigInt         *fine_to_coarse_offd = NULL;
   HYPRE_Int            *coarse_counter;
   HYPRE_Int             coarse_shift;
   HYPRE_BigInt          total_global_cpts;
   HYPRE_Int             num_cols_P_offd;
   HYPRE_BigInt          my_first_cpt;

   HYPRE_Int             i, i1;
   HYPRE_Int             j, jl, jj;
   HYPRE_Int             start;

   HYPRE_Int             my_id;
   HYPRE_Int             num_procs;
   HYPRE_Int             num_threads;
   HYPRE_Int             num_sends;
   HYPRE_Int             index;
   HYPRE_Int             ns, ne, size, rest;
   HYPRE_Int            *int_buf_data = NULL;
   HYPRE_BigInt         *big_buf_data = NULL;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block_p;
   HYPRE_Real       *sum_block_n;
   HYPRE_Real       *r_block;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = hypre_NumThreads();

   if (num_functions > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for num_functions > 1!");
   }

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/


   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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

      /* loop over the fine grid points */
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector (fine_to_coarse is the mapping vector).
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] > 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] > 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   zero_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }
   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block_p =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block_n =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   r_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
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
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         big_buf_data[index++] = my_first_cpt
                                 + fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data, fine_to_coarse_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }

      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }

      for (i = ns; i < ne; i++)
      {

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            /* P_diag_data[jj_counter] = one; */
            hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                              &P_diag_data[jj_counter * bnnz],
                                              1.0, block_size);
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);
                  jj_counter++;
               }

            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     /* P_offd_data[jj_counter_offd] = zero; */
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);
                     jj_counter_offd++;

                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;
            /* get the diagonal block */
            /* diagonal = A_diag_data[A_diag_i[i]];*/
            hypre_CSRBlockMatrixBlockCopyData(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block,
                                              1.0, block_size);


            /* Loop over ith row of A.  First, the diagonal part of A */
            /*sum_N_pos = 0;
              sum_N_neg = 0;
              sum_P_pos = 0;
              sum_P_neg = 0;*/
            hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block_p, 1.0,
                                              block_size);
            hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block_n, 1.0,
                                              block_size);


            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*if (A_diag_data[jj] > 0)
                 sum_N_pos += A_diag_data[jj];
                 else
                 sum_N_neg += A_diag_data[jj];*/

               hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                      sum_block_n, block_size);

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {


                  /* P_diag_data[P_marker[i1]] += A_diag_data[jj];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         &P_diag_data[P_marker[i1]*bnnz],
                                                         block_size);


                  /*if (A_diag_data[jj] > 0)
                    sum_P_pos += A_diag_data[jj];
                    else
                    sum_P_neg += A_diag_data[jj];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj * bnnz],
                                                         sum_block_p, block_size);

               }
            }

            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*if (A_offd_data[jj] > 0)
                    sum_N_pos += A_offd_data[jj];
                    else
                    sum_N_neg += A_offd_data[jj];*/
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj * bnnz],
                                                         sum_block_n, block_size);

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];*/
                     hypre_CSRBlockMatrixBlockAddAccumulate( &A_offd_data[jj * bnnz],
                                                             &P_offd_data[P_marker_offd[i1]*bnnz],
                                                             block_size);
                     /*if (A_offd_data[jj] > 0)
                       sum_P_pos += A_offd_data[jj];
                       else
                       sum_P_neg += A_offd_data[jj];*/
                     hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj * bnnz],
                                                            sum_block_p, block_size);

                  }
               }
            }


            /*if (sum_P_neg) alfa = sum_N_neg/sum_P_neg/diagonal;
              if (sum_P_pos) beta = sum_N_pos/sum_P_pos/diagonal;*/

            /*r_block = sum_block_n*sum_block_p^-1*/
            hypre_CSRBlockMatrixBlockMultInv(sum_block_p, sum_block_n,
                                             r_block, block_size);

            /* sum_block_n= diagonal^-1*r_block */
            hypre_CSRBlockMatrixBlockInvMult(diagonal_block, r_block,
                                             sum_block_n, block_size);

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               /*if (P_diag_data[jj]> 0)
                 P_diag_data[jj] *= -beta;
                 else
                 P_diag_data[jj] *= -alfa;*/

               hypre_CSRBlockMatrixBlockCopyData( &P_diag_data[jj * bnnz],
                                                  r_block, -1.0, block_size);


               hypre_CSRBlockMatrixBlockMultAdd(sum_block_n, r_block, 0.0,
                                                &P_diag_data[jj * bnnz], block_size);
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               /*if (P_offd_data[jj]> 0)
                 P_offd_data[jj] *= -beta;
                 else
                 P_offd_data[jj] *= -alfa;*/

               hypre_CSRBlockMatrixBlockCopyData( &P_offd_data[jj * bnnz],
                                                  r_block, -1.0, block_size);

               hypre_CSRBlockMatrixBlockMultAdd(sum_block_n, r_block, 0.0,
                                                &P_offd_data[jj * bnnz], block_size);
            }
         }

         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                     hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                     total_global_cpts,
                                     hypre_ParCSRBlockMatrixColStarts(A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine]);

   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   if (num_cols_P_offd)
   {
      hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P, A, tmp_map_offd, fine_to_coarse_offd);

   *P_ptr = P;

   hypre_TFree(zero_block, HYPRE_MEMORY_HOST);
   hypre_TFree(identity_block, HYPRE_MEMORY_HOST);
   hypre_TFree(diagonal_block, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block_n, HYPRE_MEMORY_HOST);
   hypre_TFree(sum_block_p, HYPRE_MEMORY_HOST);
   hypre_TFree(r_block, HYPRE_MEMORY_HOST);


   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return (0);

}

#if 0  /* not finished yet! */

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildBlockStdInterp
 *  Comment: The interpolatory weighting can be changed with the sep_weight
 *           variable. This can enable not separating negative and positive
 *           off diagonals in the weight formula.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildBlockStdInterp(hypre_ParCSRBlockMatrix *A,
                                   HYPRE_Int *CF_marker,
                                   hypre_ParCSRMatrix   *S,
                                   HYPRE_Int *num_cpts_global,
                                   HYPRE_Int num_functions,
                                   HYPRE_Int *dof_func,
                                   HYPRE_Int debug_flag,
                                   HYPRE_Real    trunc_factor,
                                   HYPRE_Int max_elmts,
                                   HYPRE_Int sep_weight,
                                   hypre_ParCSRBlockMatrix  **P_ptr)
{
   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRBlockMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   HYPRE_Int              my_id, num_procs;

   /* Variables to store input variables */
   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);

   HYPRE_Int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   HYPRE_Int             *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);
   HYPRE_Int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_Int              col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   HYPRE_Int              local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   HYPRE_Int              col_n = col_1 + local_numrows;
   HYPRE_Int              total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRBlockMatrix *P;
   hypre_CSRBlockMatrix    *P_diag;
   hypre_CSRBlockMatrix    *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int             *P_diag_i, *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int             *P_offd_i, *P_offd_j;

   HYPRE_Int               *col_map_offd_P;
   HYPRE_Int              P_diag_size;
   HYPRE_Int              P_offd_size;
   HYPRE_Int             *P_marker;
   HYPRE_Int             *P_marker_offd = NULL;
   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *tmp_CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRBlockMatrix *A_ext;
   HYPRE_Real      *A_ext_data;
   HYPRE_Int             *A_ext_i;
   HYPRE_Int             *A_ext_j;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd = NULL;
   HYPRE_Int             *found;

   HYPRE_Int              num_cols_P_offd;
   HYPRE_Int              newoff, loc_col;
   HYPRE_Int              A_ext_rows, full_off_procNodes;

   hypre_CSRMatrix *Sop;
   HYPRE_Int             *Sop_i;
   HYPRE_Int             *Sop_j;

   HYPRE_Int              Soprows;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int              jj_begin_row, jj_end_row;
   HYPRE_Int              jj_begin_row_offd = 0;
   HYPRE_Int              jj_end_row_offd = 0;
   HYPRE_Int              coarse_counter, coarse_counter_offd;
   HYPRE_Int             *ihat, *ihat_offd = NULL;
   HYPRE_Int             *ipnt, *ipnt_offd = NULL;
   HYPRE_Int              strong_f_marker = -2;

   /* Interpolation weight variables */
   HYPRE_Real      *ahat, *ahat_offd = NULL;
   HYPRE_Real       sum_pos, sum_pos_C, sum_neg, sum_neg_C, sum, sum_C;
   HYPRE_Real       diagonal, distribute;
   HYPRE_Real       alfa, beta;

   /* Loop variables */
   HYPRE_Int              index;
   HYPRE_Int              start_indexing = 0;
   HYPRE_Int              i, i1, j, j1, jj, kk, k1;
   HYPRE_Int              cnt_c, cnt_f, cnt_c_offd, cnt_f_offd, indx;

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   HYPRE_Real       wall_time;
   HYPRE_Real       wall_1 = 0;
   HYPRE_Real       wall_2 = 0;
   HYPRE_Real       wall_3 = 0;


   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   HYPRE_Real       *identity_block;
   HYPRE_Real       *zero_block;
   HYPRE_Real       *diagonal_block;
   HYPRE_Real       *sum_block;
   HYPRE_Real       *distribute_block;

   HYPRE_Int                  block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int                  bnnz = block_size * block_size;



   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      /*----------------------------------------------------------------------
       * Get the off processors rows for A and S, associated with columns in
       * A_offd and S_offd.
       *---------------------------------------------------------------------*/
      A_ext         = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i       = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j       = hypre_CSRBlockMatrixJ(A_ext);
      A_ext_data    = hypre_CSRBlockMatrixData(A_ext);
      A_ext_rows    = hypre_CSRBlockMatrixNumRows(A_ext);


      /* FIX THIS! - Sop - block or ???*/

      Sop           = hypre_ParCSRMatrixExtractBExt(S, A, 0);
      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixJ(Sop);
      Soprows       = hypre_CSRMatrixNumRows(Sop);

      /* Find nodes that are neighbors of neighbors, not found in offd */
      newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j,
                              Soprows, col_map_offd, col_1, col_n,
                              Sop_i, Sop_j, CF_marker, comm_pkg);
      if (newoff >= 0)
      {
         full_off_procNodes = newoff + num_cols_A_offd;
      }
      else
      {
         return (1);
      }

      /* Possibly add new points and new processors to the comm_pkg, all
       * processors need new_comm_pkg */

      /* AHB - create a new comm package just for extended info -
         this will work better with the assumed partition*/

      /* FIX THIS: Block version of this? */
      hypre_ParCSRFindExtendCommPkg(A, newoff, found,
                                    &extend_comm_pkg);

      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);

      if (num_functions > 1 && full_off_procNodes > 0)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      }

      alt_insert_new_nodes(comm_pkg, extend_comm_pkg, CF_marker,
                           full_off_procNodes, CF_marker_offd);

      if (num_functions > 1)
      {
         alt_insert_new_nodes(comm_pkg, extend_comm_pkg, dof_func,
                              full_off_procNodes, dof_func_offd);
      }
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);

   P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);


   /* FIX THIS - figure out sizes - need bnnz? */
   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                   fine_to_coarse_offd, P_marker, P_marker_offd,
                   tmp_CF_marker_offd);


   /* stuff for blocks */
   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   zero_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   for (i = 0; i < block_size; i++)
   {
      identity_block[i * block_size + i] = 1.0;
   }
   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   sum_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);
   distribute_block =  hypre_CTAlloc(HYPRE_Real,  bnnz, HYPRE_MEMORY_HOST);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else if (CF_marker[i] != -3)
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];
            if (CF_marker[i1] >= 0)
            {
               /* i1 is a C point */
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
            else if (CF_marker[i1] != -3)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < P_diag_i[i])
                     {
                        P_marker[k1] = jj_counter;
                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < P_offd_i[i])
                        {
                           tmp_CF_marker_offd[k1] = 1;
                           P_marker_offd[k1] = jj_counter_offd;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                   * numbers and entries that could be in S_diag or S_offd or
                   * neither. */
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     k1 = Sop_j[kk];
                     if (k1 >= col_1 && k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = k1 - col_1;
                        if (CF_marker[loc_col] >= 0)
                        {
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                           }
                        }
                     }
                     else
                     {
                        loc_col = -k1 - 1;
                        if (CF_marker_offd[loc_col] >= 0)
                        {
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     determine structure    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/


   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_HOST) * bnnz;

   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size * bnnz, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      for (i = 0; i < n_fine; i++)
      {
         fine_to_coarse[i] += my_first_cpt;
      }

      alt_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                           full_off_procNodes,
                           fine_to_coarse_offd);

      for (i = 0; i < n_fine; i++)
      {
         fine_to_coarse[i] -= my_first_cpt;
      }
   }

   /* Initialize ahat, which is a modification to a, used in the standard
    * interpolation routine. */
   ahat = hypre_CTAlloc(HYPRE_Real,  n_fine * bnnz, HYPRE_MEMORY_HOST); /* this is data array */
   ihat = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   ipnt = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);


   if (full_off_procNodes)
   {
      ahat_offd = hypre_CTAlloc(HYPRE_Real,  full_off_procNodes * bnnz,
                                HYPRE_MEMORY_HOST);  /* this is data array */
      ihat_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      ipnt_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
      ahat[i] = 0;
      ihat[i] = -1;
   }
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
      ahat_offd[i] = 0;
      ihat_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];


         /* P_diag_data[jj_counter] = one; */
         hypre_CSRBlockMatrixBlockCopyData(identity_block,
                                           &P_diag_data[jj_counter * bnnz],
                                           1.0, block_size);

         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }
         strong_f_marker--;
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = i1;
                  /* P_diag_data[jj_counter] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                    &P_diag_data[jj_counter * bnnz],
                                                    1.0, block_size);

                  jj_counter++;
               }
            }
            else  if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < jj_begin_row)
                     {
                        P_marker[k1] = jj_counter;
                        P_diag_j[jj_counter] = k1;
                        /* P_diag_data[jj_counter] = zero; */
                        hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                          &P_diag_data[jj_counter * bnnz],
                                                          1.0, block_size);

                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < jj_begin_row_offd)
                        {
                           P_marker_offd[k1] = jj_counter_offd;
                           P_offd_j[jj_counter_offd] = k1;
                           /* P_offd_data[jj_counter_offd] = zero; */
                           hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                             &P_offd_data[jj_counter_offd * bnnz],
                                                             1.0, block_size);

                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }

         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     /* P_offd_data[jj_counter_offd] = zero;*/
                     hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                       &P_offd_data[jj_counter_offd * bnnz],
                                                       1.0, block_size);

                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     k1 = Sop_j[kk];
                     if (k1 >= col_1 && k1 < col_n)
                     {
                        loc_col = k1 - col_1;
                        if (CF_marker[loc_col] >= 0)
                        {
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = loc_col;
                              /* P_diag_data[jj_counter] = zero;*/
                              hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                                &P_diag_data[jj_counter * bnnz],
                                                                1.0, block_size);
                              jj_counter++;
                           }
                        }
                     }
                     else
                     {
                        loc_col = -k1 - 1;
                        if (CF_marker_offd[loc_col] >= 0)
                        {
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              /* P_offd_data[jj_counter_offd] = zero;*/
                              hypre_CSRBlockMatrixBlockCopyData(zero_block,
                                                                &P_offd_data[jj_counter_offd * bnnz],
                                                                1.0, block_size);
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }

         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_1 += wall_time;
            fflush(NULL);
         }

         /* FIX THIS - is a_hat  - need to copy block data to ahat */

         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }
         cnt_c = 0;
         cnt_f = jj_end_row - jj_begin_row;
         cnt_c_offd = 0;
         cnt_f_offd = jj_end_row_offd - jj_begin_row_offd;
         ihat[i] = cnt_f;
         ipnt[cnt_f] = i;
         ahat[cnt_f++] = A_diag_data[A_diag_i[i]];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is direct neighbor */
            i1 = A_diag_j[jj];
            if (P_marker[i1] != strong_f_marker)
            {
               indx = ihat[i1];
               if (indx > -1)
               {
                  ahat[indx] += A_diag_data[jj];
               }
               else if (P_marker[i1] >= jj_begin_row)
               {
                  ihat[i1] = cnt_c;
                  ipnt[cnt_c] = i1;
                  ahat[cnt_c++] += A_diag_data[jj];
               }
               else if (CF_marker[i1] != -3)
               {
                  ihat[i1] = cnt_f;
                  ipnt[cnt_f] = i1;
                  ahat[cnt_f++] += A_diag_data[jj];
               }
            }
            else
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  distribute = A_diag_data[jj] / A_diag_data[A_diag_i[i1]];
                  for (kk = A_diag_i[i1] + 1; kk < A_diag_i[i1 + 1]; kk++)
                  {
                     k1 = A_diag_j[kk];
                     indx = ihat[k1];
                     if (indx > -1)
                     {
                        ahat[indx] -= A_diag_data[kk] * distribute;
                     }
                     else if (P_marker[k1] >= jj_begin_row)
                     {
                        ihat[k1] = cnt_c;
                        ipnt[cnt_c] = k1;
                        ahat[cnt_c++] -= A_diag_data[kk] * distribute;
                     }
                     else
                     {
                        ihat[k1] = cnt_f;
                        ipnt[cnt_f] = k1;
                        ahat[cnt_f++] -= A_diag_data[kk] * distribute;
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = A_offd_i[i1]; kk < A_offd_i[i1 + 1]; kk++)
                     {
                        k1 = A_offd_j[kk];
                        indx = ihat_offd[k1];
                        if (num_functions == 1 || dof_func[i1] == dof_func_offd[k1])
                        {
                           if (indx > -1)
                           {
                              ahat_offd[indx] -= A_offd_data[kk] * distribute;
                           }
                           else if (P_marker_offd[k1] >= jj_begin_row_offd)
                           {
                              ihat_offd[k1] = cnt_c_offd;
                              ipnt_offd[cnt_c_offd] = k1;
                              ahat_offd[cnt_c_offd++] -= A_offd_data[kk] * distribute;
                           }
                           else
                           {
                              ihat_offd[k1] = cnt_f_offd;
                              ipnt_offd[cnt_f_offd] = k1;
                              ahat_offd[cnt_f_offd++] -= A_offd_data[kk] * distribute;
                           }
                        }
                     }
                  }
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] != strong_f_marker)
               {
                  indx = ihat_offd[i1];
                  if (indx > -1)
                  {
                     ahat_offd[indx] += A_offd_data[jj];
                  }
                  else if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     ihat_offd[i1] = cnt_c_offd;
                     ipnt_offd[cnt_c_offd] = i1;
                     ahat_offd[cnt_c_offd++] += A_offd_data[jj];
                  }
                  else if (CF_marker_offd[i1] != -3)
                  {
                     ihat_offd[i1] = cnt_f_offd;
                     ipnt_offd[cnt_f_offd] = i1;
                     ahat_offd[cnt_f_offd++] += A_offd_data[jj];
                  }
               }
               else
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     distribute = A_offd_data[jj] / A_ext_data[A_ext_i[i1]];
                     for (kk = A_ext_i[i1] + 1; kk < A_ext_i[i1 + 1]; kk++)
                     {
                        k1 = A_ext_j[kk];
                        if (k1 >= col_1 && k1 < col_n)
                        {
                           /*diag*/
                           loc_col = k1 - col_1;
                           indx = ihat[loc_col];
                           if (indx > -1)
                           {
                              ahat[indx] -= A_ext_data[kk] * distribute;
                           }
                           else if (P_marker[loc_col] >= jj_begin_row)
                           {
                              ihat[loc_col] = cnt_c;
                              ipnt[cnt_c] = loc_col;
                              ahat[cnt_c++] -= A_ext_data[kk] * distribute;
                           }
                           else
                           {
                              ihat[loc_col] = cnt_f;
                              ipnt[cnt_f] = loc_col;
                              ahat[cnt_f++] -= A_ext_data[kk] * distribute;
                           }
                        }
                        else
                        {
                           loc_col = -k1 - 1;
                           if (num_functions == 1 ||
                               dof_func_offd[loc_col] == dof_func_offd[i1])
                           {
                              indx = ihat_offd[loc_col];
                              if (indx > -1)
                              {
                                 ahat_offd[indx] -= A_ext_data[kk] * distribute;
                              }
                              else if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                              {
                                 ihat_offd[loc_col] = cnt_c_offd;
                                 ipnt_offd[cnt_c_offd] = loc_col;
                                 ahat_offd[cnt_c_offd++] -= A_ext_data[kk] * distribute;
                              }
                              else
                              {
                                 ihat_offd[loc_col] = cnt_f_offd;
                                 ipnt_offd[cnt_f_offd] = loc_col;
                                 ahat_offd[cnt_f_offd++] -= A_ext_data[kk] * distribute;
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_2 += wall_time;
            fflush(NULL);
         }

         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }
         diagonal = ahat[cnt_c];
         ahat[cnt_c] = 0;
         sum_pos = 0;
         sum_pos_C = 0;
         sum_neg = 0;
         sum_neg_C = 0;
         sum = 0;
         sum_C = 0;
         if (sep_weight == 1)
         {
            for (jj = 0; jj < cnt_c; jj++)
            {
               if (ahat[jj] > 0)
               {
                  sum_pos_C += ahat[jj];
               }
               else
               {
                  sum_neg_C += ahat[jj];
               }
            }
            if (num_procs > 1)
            {
               for (jj = 0; jj < cnt_c_offd; jj++)
               {
                  if (ahat_offd[jj] > 0)
                  {
                     sum_pos_C += ahat_offd[jj];
                  }
                  else
                  {
                     sum_neg_C += ahat_offd[jj];
                  }
               }
            }
            sum_pos = sum_pos_C;
            sum_neg = sum_neg_C;
            for (jj = cnt_c + 1; jj < cnt_f; jj++)
            {
               if (ahat[jj] > 0)
               {
                  sum_pos += ahat[jj];
               }
               else
               {
                  sum_neg += ahat[jj];
               }
               ahat[jj] = 0;
            }
            if (num_procs > 1)
            {
               for (jj = cnt_c_offd; jj < cnt_f_offd; jj++)
               {
                  if (ahat_offd[jj] > 0)
                  {
                     sum_pos += ahat_offd[jj];
                  }
                  else
                  {
                     sum_neg += ahat_offd[jj];
                  }
                  ahat_offd[jj] = 0;
               }
            }
            if (sum_neg_C) { alfa = sum_neg / sum_neg_C / diagonal; }
            if (sum_pos_C) { beta = sum_pos / sum_pos_C / diagonal; }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               j1 = ihat[P_diag_j[jj]];
               if (ahat[j1] > 0)
               {
                  P_diag_data[jj] = -beta * ahat[j1];
               }
               else
               {
                  P_diag_data[jj] = -alfa * ahat[j1];
               }

               P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
               ahat[j1] = 0;
            }
            for (jj = 0; jj < cnt_f; jj++)
            {
               ihat[ipnt[jj]] = -1;
            }
            if (num_procs > 1)
            {
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  j1 = ihat_offd[P_offd_j[jj]];
                  if (ahat_offd[j1] > 0)
                  {
                     P_offd_data[jj] = -beta * ahat_offd[j1];
                  }
                  else
                  {
                     P_offd_data[jj] = -alfa * ahat_offd[j1];
                  }

                  ahat_offd[j1] = 0;
               }
               for (jj = 0; jj < cnt_f_offd; jj++)
               {
                  ihat_offd[ipnt_offd[jj]] = -1;
               }
            }
         }
         else
         {
            for (jj = 0; jj < cnt_c; jj++)
            {
               sum_C += ahat[jj];
            }
            if (num_procs > 1)
            {
               for (jj = 0; jj < cnt_c_offd; jj++)
               {
                  sum_C += ahat_offd[jj];
               }
            }
            sum = sum_C;
            for (jj = cnt_c + 1; jj < cnt_f; jj++)
            {
               sum += ahat[jj];
               ahat[jj] = 0;
            }
            if (num_procs > 1)
            {
               for (jj = cnt_c_offd; jj < cnt_f_offd; jj++)
               {
                  sum += ahat_offd[jj];
                  ahat_offd[jj] = 0;
               }
            }
            if (sum_C) { alfa = sum / sum_C / diagonal; }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               j1 = ihat[P_diag_j[jj]];
               P_diag_data[jj] = -alfa * ahat[j1];
               P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
               ahat[j1] = 0;
            }
            for (jj = 0; jj < cnt_f; jj++)
            {
               ihat[ipnt[jj]] = -1;
            }
            if (num_procs > 1)
            {
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  j1 = ihat_offd[P_offd_j[jj]];
                  P_offd_data[jj] = -alfa * ahat_offd[j1];
                  ahat_offd[j1] = 0;
               }
               for (jj = 0; jj < cnt_f_offd; jj++)
               {
                  ihat_offd[ipnt_offd[jj]] = -1;
               }
            }
         }
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_3 += wall_time;
            fflush(NULL);
         }
      }
   }

   if (debug_flag == 4)
   {
      hypre_printf("Proc = %d fill part 1 %f part 2 %f  part 3 %f\n",
                   my_id, wall_1, wall_2, wall_3);
      fflush(NULL);
   }
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      P_marker = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);

      for (i = 0; i < full_off_procNodes; i++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            if (tmp_CF_marker_offd[index] >= 0)
            {
               num_cols_P_offd++;
               P_marker[index] = 1;
            }
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int,  num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while ( P_marker[index] == 0) { index++; }
         col_map_offd_P[i] = index++;
      }
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
                                          P_offd_j[i],
                                          num_cols_P_offd);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }

         col_map_offd_P[i] = fine_to_coarse_offd[index];
         index++;
      }

      /* Sort the col_map_offd_P and P_offd_j correctly */
      for (i = 0; i < num_cols_P_offd; i++)
      {
         P_marker[i] = col_map_offd_P[i];
      }

      /* Check if sort actually changed anything */
      if (ssort(col_map_offd_P, num_cols_P_offd))
      {
         for (i = 0; i < P_offd_size; i++)
            for (j = 0; j < num_cols_P_offd; j++)
               if (P_marker[P_offd_j[i]] == col_map_offd_P[j])
               {
                  P_offd_j[i] = j;
                  j = num_cols_P_offd;
               }
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(ahat, HYPRE_MEMORY_HOST);
   hypre_TFree(ihat, HYPRE_MEMORY_HOST);
   hypre_TFree(ipnt, HYPRE_MEMORY_HOST);

   if (full_off_procNodes)
   {
      hypre_TFree(ahat_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(ihat_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(ipnt_offd, HYPRE_MEMORY_HOST);
   }
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);

      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }
      hypre_TFree(found, HYPRE_MEMORY_HOST);

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);

   }


   return hypre_error_flag;
}

#endif

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

__global__ void hypre_BoomerAMGCreateS_rowcount( HYPRE_Int nr_of_rows,
                                                 HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
                                                 HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
                                                 HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                                 HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
                                                 HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
                                                 HYPRE_Int* jS_diag, HYPRE_Int* jS_offd );
__global__ void hypre_BoomerAMGCreateSabs_rowcount( HYPRE_Int nr_of_rows,
                                                    HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
                                                    HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
                                                    HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                                    HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
                                                    HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
                                                    HYPRE_Int* jS_diag, HYPRE_Int* jS_offd );


/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
HYPRE_Int
hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix    *A,
                             HYPRE_Int              abs_soc,
                             HYPRE_Real             strength_threshold,
                             HYPRE_Real             max_row_sum,
                             HYPRE_Int              num_functions,
                             HYPRE_Int             *dof_func,
                             hypre_ParCSRMatrix   **S_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int               *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real              *A_diag_data     = hypre_CSRMatrixData(A_diag);
   hypre_CSRMatrix         *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int               *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real              *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt            *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int                num_nonzeros_diag;
   HYPRE_Int                num_nonzeros_offd;
   HYPRE_Int                num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   hypre_ParCSRMatrix      *S;
   hypre_CSRMatrix         *S_diag;
   HYPRE_Int               *S_diag_i;
   HYPRE_Int               *S_diag_j, *S_temp_diag_j;
   /* HYPRE_Real           *S_diag_data; */
   hypre_CSRMatrix         *S_offd;
   HYPRE_Int               *S_offd_i = NULL;
   HYPRE_Int               *S_offd_j = NULL, *S_temp_offd_j = NULL;
   /* HYPRE_Real           *S_offd_data; */
   HYPRE_Int                i;
   HYPRE_Int                ierr = 0;
   HYPRE_Int               *dof_func_offd_dev = NULL;
   HYPRE_Int               *dof_func_dev = NULL;
   HYPRE_Int                num_sends;
   HYPRE_Int                index, start, j;

   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * Default "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * If abs_soc != 0, then use an absolute strength of connection:
    * i depends on j if
    *     abs(aij) > hypre_max (k != i) abs(aik)
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(A_diag);
   num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(A_offd);

   S_diag_i = hypre_TAlloc(HYPRE_Int, num_variables+1, memory_location);
   S_offd_i = hypre_TAlloc(HYPRE_Int, num_variables+1, memory_location);
   S_temp_diag_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_DEVICE);
   S_temp_offd_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_offd, HYPRE_MEMORY_DEVICE);

   if (num_functions > 1)
   {
      dof_func_offd_dev = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);
   }

  /*-------------------------------------------------------------------
    * Get the dof_func data for the off-processor columns
    *-------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_functions > 1)
   {
      HYPRE_Int *int_buf_data = hypre_TAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends), HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, HYPRE_MEMORY_HOST, int_buf_data,
                                                    HYPRE_MEMORY_DEVICE, dof_func_offd_dev);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);

      dof_func_dev = hypre_TAlloc(HYPRE_Int, num_variables, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(dof_func_dev, dof_func, HYPRE_Int, num_variables, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }

   /* count the row nnz of S */
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(num_variables, "warp", bDim);

   if (abs_soc)
   {
      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGCreateSabs_rowcount, gDim, bDim,
                         num_variables, max_row_sum, strength_threshold,
                         A_diag_data, A_diag_i, A_diag_j,
                         A_offd_data, A_offd_i, A_offd_j,
                         S_temp_diag_j, S_temp_offd_j,
                         num_functions, dof_func_dev, dof_func_offd_dev,
                         S_diag_i, S_offd_i );
   }
   else
   {
      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGCreateS_rowcount, gDim, bDim,
                         num_variables, max_row_sum, strength_threshold,
                         A_diag_data, A_diag_i, A_diag_j,
                         A_offd_data, A_offd_i, A_offd_j,
                         S_temp_diag_j, S_temp_offd_j,
                         num_functions, dof_func_dev, dof_func_offd_dev,
                         S_diag_i, S_offd_i );
   }

   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_diag_i);
   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_offd_i);

   HYPRE_Int *tmp, S_num_nonzeros_diag, S_num_nonzeros_offd;

   hypre_TMemcpy(&S_num_nonzeros_diag, &S_diag_i[num_variables], HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);
   hypre_TMemcpy(&S_num_nonzeros_offd, &S_offd_i[num_variables], HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);

   S_diag_j = hypre_TAlloc(HYPRE_Int, S_num_nonzeros_diag, memory_location);
   S_offd_j = hypre_TAlloc(HYPRE_Int, S_num_nonzeros_offd, memory_location);

   tmp = HYPRE_THRUST_CALL(copy_if, S_temp_diag_j, S_temp_diag_j + num_nonzeros_diag, S_diag_j, is_nonnegative<HYPRE_Int>());

   hypre_assert(S_num_nonzeros_diag == tmp - S_diag_j);

   tmp = HYPRE_THRUST_CALL(copy_if, S_temp_offd_j, S_temp_offd_j + num_nonzeros_offd, S_offd_j, is_nonnegative<HYPRE_Int>());

   hypre_assert(S_num_nonzeros_offd == tmp - S_offd_j);

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars, row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);

   hypre_CSRMatrixNumNonzeros(S_diag) = S_num_nonzeros_diag;
   hypre_CSRMatrixNumNonzeros(S_offd) = S_num_nonzeros_offd;
   hypre_CSRMatrixI(S_diag) = S_diag_i;
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixI(S_offd) = S_offd_i;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;
   hypre_CSRMatrixMemoryLocation(S_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_offd) = memory_location;

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   hypre_ParCSRMatrixColMapOffd(S) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(S), hypre_ParCSRMatrixColMapOffd(A),
                 HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixSocDiagJ(S) = S_temp_diag_j;
   hypre_ParCSRMatrixSocOffdJ(S) = S_temp_offd_j;

   *S_ptr = S;

   hypre_TFree(dof_func_offd_dev, HYPRE_MEMORY_DEVICE);
   hypre_TFree(dof_func_dev,      HYPRE_MEMORY_DEVICE);
   /*
   hypre_TFree(S_temp_diag_j,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(S_temp_offd_j,     HYPRE_MEMORY_DEVICE);
   */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] += hypre_MPI_Wtime();
#endif

   return (ierr);
}

/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGCreateS_rowcount( HYPRE_Int   nr_of_rows,
                                                  HYPRE_Real  max_row_sum,
                                                  HYPRE_Real  strength_threshold,
                                                  HYPRE_Real *A_diag_data,
                                                  HYPRE_Int  *A_diag_i,
                                                  HYPRE_Int  *A_diag_j,
                                                  HYPRE_Real *A_offd_data,
                                                  HYPRE_Int  *A_offd_i,
                                                  HYPRE_Int  *A_offd_j,
                                                  HYPRE_Int  *S_temp_diag_j,
                                                  HYPRE_Int  *S_temp_offd_j,
                                                  HYPRE_Int   num_functions,
                                                  HYPRE_Int  *dof_func,
                                                  HYPRE_Int  *dof_func_offd,
                                                  HYPRE_Int  *jS_diag,
                                                  HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale = 0.0, row_sum = 0.0, row_max = 0.0, row_min = 0.0, diag = 0.0;
   HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p_diag, q_diag, p_offd, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
   q_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (HYPRE_Int i = p_diag + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_diag); i += HYPRE_WARP_SIZE)
   {
      if (i < q_diag)
      {
         const HYPRE_Int col = read_only_load(&A_diag_j[i]);

         if ( num_functions == 1 || row == col ||
              read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
         {
            const HYPRE_Real v = read_only_load(&A_diag_data[i]);
            row_sum += v;
            if (row == col)
            {
               diag = v;
               diag_pos = i;
            }
            else
            {
               row_max = hypre_max(row_max, v);
               row_min = hypre_min(row_min, v);
            }
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
   q_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int i = p_offd + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_offd); i += HYPRE_WARP_SIZE)
   {
      if (i < q_offd)
      {
         if ( num_functions == 1 ||
              read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
         {
            const HYPRE_Real v = read_only_load(&A_offd_data[i]);
            row_sum += v;
            row_max = hypre_max(row_max, v);
            row_min = hypre_min(row_min, v);
         }
      }
   }

   diag = warp_allreduce_sum(diag);

   /* sign of diag */
   const HYPRE_Int sdiag = diag > 0.0 ? 1 : -1;

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(row_sum);

   if (diag > 0.0)
   {
      row_scale = warp_allreduce_min(row_min);
   }
   else
   {
      row_scale = warp_allreduce_max(row_max);
   }

   /* compute row of S */
   HYPRE_Int all_weak = max_row_sum < 1.0 && fabs(row_sum) > fabs(diag) * max_row_sum;
   const HYPRE_Real thresh = sdiag * strength_threshold * row_scale;

   for (HYPRE_Int i = p_diag + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_diag); i += HYPRE_WARP_SIZE)
   {
      if (i < q_diag)
      {
         const HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                                ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                                                        read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                                sdiag * read_only_load(&A_diag_data[i]) < thresh;
         S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
         row_nnz_diag += cond;
      }
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (HYPRE_Int i = p_offd + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_offd); i += HYPRE_WARP_SIZE)
   {
      if (i < q_offd)
      {
         const HYPRE_Int cond = all_weak == 0 &&
                                ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                                                        read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                                sdiag * read_only_load(&A_offd_data[i]) < thresh;
         S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
         row_nnz_offd += cond;
      }
   }

   row_nnz_diag = warp_reduce_sum(row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd);

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

HYPRE_Int
hypre_BoomerAMGMakeSocFromSDevice( hypre_ParCSRMatrix *A,
                                   hypre_ParCSRMatrix *S)
{
   if (!hypre_ParCSRMatrixSocDiagJ(S))
   {
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
      HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(A_diag);
      HYPRE_Int *soc_diag = hypre_TAlloc(HYPRE_Int, nnz_diag, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixIntersectPattern(A_diag, S_diag, soc_diag, 1);
      hypre_ParCSRMatrixSocDiagJ(S) = soc_diag;
   }

   if (!hypre_ParCSRMatrixSocOffdJ(S))
   {
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
      HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(A_offd);
      HYPRE_Int *soc_offd = hypre_TAlloc(HYPRE_Int, nnz_offd, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixIntersectPattern(A_offd, S_offd, soc_offd, 0);
      hypre_ParCSRMatrixSocOffdJ(S) = soc_offd;
   }

   return hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGCreateSabs_rowcount( HYPRE_Int   nr_of_rows,
                                                     HYPRE_Real  max_row_sum,
                                                     HYPRE_Real  strength_threshold,
                                                     HYPRE_Real *A_diag_data,
                                                     HYPRE_Int  *A_diag_i,
                                                     HYPRE_Int  *A_diag_j,
                                                     HYPRE_Real *A_offd_data,
                                                     HYPRE_Int  *A_offd_i,
                                                     HYPRE_Int  *A_offd_j,
                                                     HYPRE_Int  *S_temp_diag_j,
                                                     HYPRE_Int  *S_temp_offd_j,
                                                     HYPRE_Int   num_functions,
                                                     HYPRE_Int  *dof_func,
                                                     HYPRE_Int  *dof_func_offd,
                                                     HYPRE_Int  *jS_diag,
                                                     HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale = 0.0, row_sum = 0.0, diag = 0.0;
   HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p_diag, q_diag, p_offd, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
   q_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (HYPRE_Int i = p_diag + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_diag); i += HYPRE_WARP_SIZE)
   {
      if (i < q_diag)
      {
         const HYPRE_Int col = read_only_load(&A_diag_j[i]);

         if ( num_functions == 1 || row == col ||
              read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
         {
            const HYPRE_Real v = hypre_cabs( read_only_load(&A_diag_data[i]) );
            row_sum += v;
            if (row == col)
            {
               diag = v;
               diag_pos = i;
            }
            else
            {
               row_scale = hypre_max(row_scale, v);
            }
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
   q_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int i = p_offd + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_offd); i += HYPRE_WARP_SIZE)
   {
      if (i < q_offd)
      {
         if ( num_functions == 1 ||
              read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
         {
            const HYPRE_Real v = hypre_cabs( read_only_load(&A_offd_data[i]) );
            row_sum += v;
            row_scale = hypre_max(row_scale, v);
         }
      }
   }

   diag = warp_allreduce_sum(diag);

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(row_sum);
   row_scale = warp_allreduce_max(row_scale);

   /* compute row of S */
   HYPRE_Int all_weak = max_row_sum < 1.0 && fabs(row_sum) < fabs(diag) * (2.0 - max_row_sum);
   const HYPRE_Real thresh = strength_threshold * row_scale;

   for (HYPRE_Int i = p_diag + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_diag); i += HYPRE_WARP_SIZE)
   {
      if (i < q_diag)
      {
         const HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                                ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                                                        read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                                hypre_cabs( read_only_load(&A_diag_data[i]) ) > thresh;
         S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
         row_nnz_diag += cond;
      }
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (HYPRE_Int i = p_offd + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q_offd); i += HYPRE_WARP_SIZE)
   {
      if (i < q_offd)
      {
         const HYPRE_Int cond = all_weak == 0 &&
                                ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                                                        read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                                hypre_cabs( read_only_load(&A_offd_data[i]) ) > thresh;
         S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
         row_nnz_offd += cond;
      }
   }

   row_nnz_diag = warp_reduce_sum(row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(row_nnz_offd);

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */

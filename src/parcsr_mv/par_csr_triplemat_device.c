/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#define PARCSRGEMM_TIMING 0

#if defined(HYPRE_USING_GPU)

/* option == 1, T = HYPRE_BigInt
 * option == 2, T = HYPRE_Int,
 */
template<HYPRE_Int option, typename T>
#if defined(HYPRE_USING_SYCL)
struct RAP_functor
#else
struct RAP_functor : public thrust::unary_function<HYPRE_Int, T>
#endif
{
   HYPRE_Int num_col;
   T         first_col;
   T        *col_map;

   RAP_functor(HYPRE_Int num_col_, T first_col_, T *col_map_)
   {
      num_col   = num_col_;
      first_col = first_col_;
      col_map   = col_map_;
   }

   __host__ __device__
   T operator()(const HYPRE_Int x) const
   {
      if (x < num_col)
      {
         if (option == 1)
         {
            return x + first_col;
         }
         else
         {
            return x;
         }
      }

      if (option == 1)
      {
         return col_map[x - num_col];
      }
      else
      {
         return col_map[x - num_col] + num_col;
      }
   }
};

/* C = A * B */
hypre_ParCSRMatrix*
hypre_ParCSRMatMatDevice( hypre_ParCSRMatrix  *A,
                          hypre_ParCSRMatrix  *B )
{
   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix    *C_diag;
   hypre_CSRMatrix    *C_offd;
   HYPRE_Int           num_cols_offd_C = 0;
   HYPRE_BigInt       *col_map_offd_C = NULL;

   HYPRE_Int num_procs;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Comm_size(comm, &num_procs);

   if ( hypre_ParCSRMatrixGlobalNumCols(A) != hypre_ParCSRMatrixGlobalNumRows(B) ||
        hypre_ParCSRMatrixNumCols(A)       != hypre_ParCSRMatrixNumRows(B) )
   {
      hypre_error_in_arg(1);
      hypre_printf(" Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

#if PARCSRGEMM_TIMING > 0
   HYPRE_Real ta, tb;
   ta = hypre_MPI_Wtime();
#endif

#if PARCSRGEMM_TIMING > 1
   HYPRE_Real t1, t2;
#endif

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/
   if (num_procs > 1)
   {
      void *request;
      hypre_CSRMatrix *Abar, *Bbar, *Cbar, *Bext;
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      /* contains communication which should be explicitly included to allow for overlap */
      hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, 1, &request);
#if PARCSRGEMM_TIMING > 1
      t2 = hypre_MPI_Wtime();
#endif
      Abar = hypre_ConcatDiagAndOffdDevice(A);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t2;
      hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif
      Bext = hypre_ParCSRMatrixExtractBExtDeviceWait(request);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1 - t2;
      hypre_ParPrintf(comm, "Time Bext %f\n", t2);
      hypre_ParPrintf(comm, "Size Bext %d %d %d\n", hypre_CSRMatrixNumRows(Bext),
                      hypre_CSRMatrixNumCols(Bext), hypre_CSRMatrixNumNonzeros(Bext));
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      hypre_ConcatDiagOffdAndExtDevice(B, Bext, &Bbar, &num_cols_offd_C, &col_map_offd_C);
      hypre_CSRMatrixDestroy(Bext);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      Cbar = hypre_CSRMatrixMultiplyDevice(Abar, Bbar);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif

      hypre_CSRMatrixDestroy(Abar);
      hypre_CSRMatrixDestroy(Bbar);

      hypre_assert(hypre_CSRMatrixNumRows(Cbar) == hypre_ParCSRMatrixNumRows(A));
      hypre_assert(hypre_CSRMatrixNumCols(Cbar) == hypre_ParCSRMatrixNumCols(B) + num_cols_offd_C);

      // split into diag and offd
#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      in_range<HYPRE_Int> pred(0, hypre_ParCSRMatrixNumCols(B) - 1);
#if defined(HYPRE_USING_SYCL)
      HYPRE_Int nnz_C_diag = HYPRE_ONEDPL_CALL( std::count_if,
                                                hypre_CSRMatrixJ(Cbar),
                                                hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                                                pred );
#else
      HYPRE_Int nnz_C_diag = HYPRE_THRUST_CALL( count_if,
                                                hypre_CSRMatrixJ(Cbar),
                                                hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                                                pred );
#endif
      HYPRE_Int nnz_C_offd = hypre_CSRMatrixNumNonzeros(Cbar) - nnz_C_diag;

      C_diag = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumRows(A), hypre_ParCSRMatrixNumCols(B),
                                     nnz_C_diag);
      hypre_CSRMatrixInitialize_v2(C_diag, 0, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *C_diag_ii = hypre_TAlloc(HYPRE_Int, nnz_C_diag, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *C_diag_j = hypre_CSRMatrixJ(C_diag);
      HYPRE_Complex *C_diag_a = hypre_CSRMatrixData(C_diag);

      HYPRE_Int *Cbar_ii = hypreDevice_CsrRowPtrsToIndices(hypre_ParCSRMatrixNumRows(A),
                                                           hypre_CSRMatrixNumNonzeros(Cbar),
                                                           hypre_CSRMatrixI(Cbar));

#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                       hypre_CSRMatrixData(Cbar)),
                                        oneapi::dpl::make_zip_iterator(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                       hypre_CSRMatrixData(Cbar)) + hypre_CSRMatrixNumNonzeros(Cbar),
                                        hypre_CSRMatrixJ(Cbar),
                                        oneapi::dpl::make_zip_iterator(C_diag_ii, C_diag_j, C_diag_a),
                                        pred );
      hypre_assert( std::get<0>(new_end.base()) == C_diag_ii + nnz_C_diag );
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                     hypre_CSRMatrixData(Cbar))),
                        thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                     hypre_CSRMatrixData(Cbar))) + hypre_CSRMatrixNumNonzeros(Cbar),
                        hypre_CSRMatrixJ(Cbar),
                        thrust::make_zip_iterator(thrust::make_tuple(C_diag_ii, C_diag_j, C_diag_a)),
                        pred );
      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_diag_ii + nnz_C_diag );
#endif
      hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(C_diag), nnz_C_diag, C_diag_ii,
                                         hypre_CSRMatrixI(C_diag));
      hypre_TFree(C_diag_ii, HYPRE_MEMORY_DEVICE);

      C_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumRows(A), num_cols_offd_C, nnz_C_offd);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *C_offd_ii = hypre_TAlloc(HYPRE_Int, nnz_C_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *C_offd_j = hypre_CSRMatrixJ(C_offd);
      HYPRE_Complex *C_offd_a = hypre_CSRMatrixData(C_offd);
#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                  hypre_CSRMatrixData(Cbar)),
                                   oneapi::dpl::make_zip_iterator(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                  hypre_CSRMatrixData(Cbar)) + hypre_CSRMatrixNumNonzeros(Cbar),
                                   hypre_CSRMatrixJ(Cbar),
                                   oneapi::dpl::make_zip_iterator(C_offd_ii, C_offd_j, C_offd_a),
                                   std::not_fn(pred) );
      hypre_assert( std::get<0>(new_end.base()) == C_offd_ii + nnz_C_offd );
#else
      new_end = HYPRE_THRUST_CALL(
                   copy_if,
                   thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                hypre_CSRMatrixData(Cbar))),
                   thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, hypre_CSRMatrixJ(Cbar),
                                                                hypre_CSRMatrixData(Cbar))) + hypre_CSRMatrixNumNonzeros(Cbar),
                   hypre_CSRMatrixJ(Cbar),
                   thrust::make_zip_iterator(thrust::make_tuple(C_offd_ii, C_offd_j, C_offd_a)),
                   thrust::not1(pred) );
      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_offd_ii + nnz_C_offd );
#endif

      hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(C_offd), nnz_C_offd, C_offd_ii,
                                         hypre_CSRMatrixI(C_offd));
      hypre_TFree(C_offd_ii, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         C_offd_j,
                         C_offd_j + nnz_C_offd,
                         C_offd_j,
      [const_val = hypre_ParCSRMatrixNumCols(B)] (const auto & x) {return x - const_val;} );
#else
      HYPRE_THRUST_CALL( transform,
                         C_offd_j,
                         C_offd_j + nnz_C_offd,
                         thrust::make_constant_iterator(hypre_ParCSRMatrixNumCols(B)),
                         C_offd_j,
                         thrust::minus<HYPRE_Int>() );
#endif

      hypre_TFree(Cbar_ii, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixDestroy(Cbar);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Split %f\n", t2);
#endif
   }
   else
   {
#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      C_diag = hypre_CSRMatrixMultiplyDevice(hypre_ParCSRMatrixDiag(A), hypre_ParCSRMatrixDiag(B));
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif
      C_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumRows(A), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(B),
                                hypre_ParCSRMatrixRowStarts(A),
                                hypre_ParCSRMatrixColStarts(B),
                                num_cols_offd_C,
                                hypre_CSRMatrixNumNonzeros(C_diag),
                                hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;
   }

   hypre_ParCSRMatrixCopyColMapOffdToHost(C);

#if PARCSRGEMM_TIMING > 0
   hypre_ForceSyncComputeStream(hypre_handle());
   tb = hypre_MPI_Wtime() - ta;
   hypre_ParPrintf(comm, "Time hypre_ParCSRMatMatDevice %f\n", tb);
#endif

   return C;
}

/* C = A^T * B */
hypre_ParCSRMatrix*
hypre_ParCSRTMatMatKTDevice( hypre_ParCSRMatrix  *A,
                             hypre_ParCSRMatrix  *B,
                             HYPRE_Int            keep_transpose)
{
   hypre_CSRMatrix *A_diag  = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd  = hypre_ParCSRMatrixOffd(A);

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix    *C_diag;
   hypre_CSRMatrix    *C_offd;
   HYPRE_Int           num_cols_offd_C = 0;
   HYPRE_BigInt       *col_map_offd_C = NULL;

   HYPRE_Int num_procs;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (hypre_ParCSRMatrixGlobalNumRows(A) != hypre_ParCSRMatrixGlobalNumRows(B) ||
       hypre_ParCSRMatrixNumRows(A)       != hypre_ParCSRMatrixNumRows(B))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

#if PARCSRGEMM_TIMING > 0
   HYPRE_Real ta, tb;
   ta = hypre_MPI_Wtime();
#endif

#if PARCSRGEMM_TIMING > 1
   HYPRE_Real t1, t2;
#endif

   if (num_procs > 1)
   {
      void *request;
      hypre_CSRMatrix *Bbar, *AbarT, *Cbar, *AT_diag, *AT_offd, *Cint, *Cext;
      hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
      HYPRE_Int local_nnz_Cbar;

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      Bbar = hypre_ConcatDiagAndOffdDevice(B);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif

      if (hypre_ParCSRMatrixDiagT(A))
      {
         AT_diag = hypre_ParCSRMatrixDiagT(A);
      }
      else
      {
         hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
      }

      if (hypre_ParCSRMatrixOffdT(A))
      {
         AT_offd = hypre_ParCSRMatrixOffdT(A);
      }
      else
      {
         hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
      }

#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Transpose %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      AbarT = hypre_CSRMatrixStack2Device(AT_diag, AT_offd);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Stack %f\n", t2);
#endif

      if (!hypre_ParCSRMatrixDiagT(A))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixDiagT(A) = AT_diag;
         }
         else
         {
            hypre_CSRMatrixDestroy(AT_diag);
         }
      }

      if (!hypre_ParCSRMatrixOffdT(A))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixOffdT(A) = AT_offd;
         }
         else
         {
            hypre_CSRMatrixDestroy(AT_offd);
         }
      }

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      Cbar = hypre_CSRMatrixMultiplyDevice(AbarT, Bbar);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif

      hypre_CSRMatrixDestroy(AbarT);
      hypre_CSRMatrixDestroy(Bbar);

      hypre_assert(hypre_CSRMatrixNumRows(Cbar) == hypre_ParCSRMatrixNumCols(A) + hypre_CSRMatrixNumCols(
                      A_offd));
      hypre_assert(hypre_CSRMatrixNumCols(Cbar) == hypre_ParCSRMatrixNumCols(B) + hypre_CSRMatrixNumCols(
                      B_offd));

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      hypre_TMemcpy(&local_nnz_Cbar, hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(A), HYPRE_Int, 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      // Cint is the bottom part of Cbar
      Cint = hypre_CSRMatrixCreate(hypre_CSRMatrixNumCols(A_offd), hypre_CSRMatrixNumCols(Cbar),
                                   hypre_CSRMatrixNumNonzeros(Cbar) - local_nnz_Cbar);
      hypre_CSRMatrixMemoryLocation(Cint) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixOwnsData(Cint) = 0;

      hypre_CSRMatrixI(Cint) = hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(A);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixI(Cint),
                         hypre_CSRMatrixI(Cint) + hypre_CSRMatrixNumRows(Cint) + 1,
                         hypre_CSRMatrixI(Cint),
      [const_val = local_nnz_Cbar] (const auto & x) {return x - const_val;} );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixI(Cint),
                         hypre_CSRMatrixI(Cint) + hypre_CSRMatrixNumRows(Cint) + 1,
                         thrust::make_constant_iterator(local_nnz_Cbar),
                         hypre_CSRMatrixI(Cint),
                         thrust::minus<HYPRE_Int>() );
#endif

      // Change Cint into a BigJ matrix
      // RL: TODO FIX the 'big' num of columns to global size
      hypre_CSRMatrixBigJ(Cint) = hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumNonzeros(Cint),
                                               HYPRE_MEMORY_DEVICE);

      RAP_functor<1, HYPRE_BigInt> func1( hypre_ParCSRMatrixNumCols(B),
                                          hypre_ParCSRMatrixFirstColDiag(B),
                                          hypre_ParCSRMatrixDeviceColMapOffd(B) );
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      hypre_CSRMatrixData(Cint) = hypre_CSRMatrixData(Cbar) + local_nnz_Cbar;

#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Cint %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      hypre_ExchangeExternalRowsDeviceInit(Cint, hypre_ParCSRMatrixCommPkg(A), 1, &request);
      Cext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_TFree(hypre_CSRMatrixBigJ(Cint), HYPRE_MEMORY_DEVICE);
      hypre_TFree(Cint, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(A), &local_nnz_Cbar, HYPRE_Int, 1,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Cext %f\n", t2);
      hypre_ParPrintf(comm, "Size Cext %d %d %d\n", hypre_CSRMatrixNumRows(Cext),
                      hypre_CSRMatrixNumCols(Cext), hypre_CSRMatrixNumNonzeros(Cext));
#endif

      /* add Cext to local part of Cbar */
      hypre_ParCSRTMatMatPartialAddDevice(hypre_ParCSRMatrixCommPkg(A),
                                          hypre_ParCSRMatrixNumCols(A),
                                          hypre_ParCSRMatrixNumCols(B),
                                          hypre_ParCSRMatrixFirstColDiag(B),
                                          hypre_ParCSRMatrixLastColDiag(B),
                                          hypre_CSRMatrixNumCols(B_offd),
                                          hypre_ParCSRMatrixDeviceColMapOffd(B),
                                          local_nnz_Cbar,
                                          Cbar,
                                          Cext,
                                          &C_diag,
                                          &C_offd,
                                          &num_cols_offd_C,
                                          &col_map_offd_C);
   }
   else
   {
      hypre_CSRMatrix *AT_diag;
      hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      hypre_CSRMatrixTransposeDevice(A_diag, &AT_diag, 1);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time Transpose %f\n", t2);
#endif
#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      C_diag = hypre_CSRMatrixMultiplyDevice(AT_diag, B_diag);
#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif
      C_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(A), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(A) = AT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(AT_diag);
      }
   }

   /* Move the diagonal entry to the first of each row */
   hypre_CSRMatrixMoveDiagFirstDevice(C_diag);

   C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumCols(A),
                                hypre_ParCSRMatrixGlobalNumCols(B),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(B),
                                num_cols_offd_C,
                                hypre_CSRMatrixNumNonzeros(C_diag),
                                hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

   hypre_ParCSRMatrixCompressOffdMapDevice(C);

   hypre_ParCSRMatrixCopyColMapOffdToHost(C);

   hypre_assert(!hypre_CSRMatrixCheckDiagFirstDevice(hypre_ParCSRMatrixDiag(C)));

   hypre_SyncComputeStream(hypre_handle());

#if PARCSRGEMM_TIMING > 0
   hypre_ForceSyncComputeStream(hypre_handle());
   tb = hypre_MPI_Wtime() - ta;
   hypre_ParPrintf(comm, "Time hypre_ParCSRTMatMatKTDevice %f\n", tb);
#endif

   return C;
}

/* C = R^{T} * A * P */
hypre_ParCSRMatrix*
hypre_ParCSRMatrixRAPKTDevice( hypre_ParCSRMatrix *R,
                               hypre_ParCSRMatrix *A,
                               hypre_ParCSRMatrix *P,
                               HYPRE_Int           keep_transpose )
{
   MPI_Comm             comm   = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix     *R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrix     *R_offd = hypre_ParCSRMatrixOffd(R);

   hypre_ParCSRMatrix  *C;
   hypre_CSRMatrix     *C_diag;
   hypre_CSRMatrix     *C_offd;
   HYPRE_Int            num_cols_offd_C = 0;
   HYPRE_BigInt        *col_map_offd_C = NULL;

   HYPRE_Int            num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);

   if ( hypre_ParCSRMatrixGlobalNumRows(R) != hypre_ParCSRMatrixGlobalNumRows(A) ||
        hypre_ParCSRMatrixGlobalNumCols(A) != hypre_ParCSRMatrixGlobalNumRows(P) )
   {
      hypre_error_in_arg(1);
      hypre_printf(" Error! Incompatible matrix global dimensions!\n");
      return NULL;
   }

   if ( hypre_ParCSRMatrixNumRows(R) != hypre_ParCSRMatrixNumRows(A) ||
        hypre_ParCSRMatrixNumCols(A) != hypre_ParCSRMatrixNumRows(P) )
   {
      hypre_error_in_arg(1);
      hypre_printf(" Error! Incompatible matrix local dimensions!\n");
      return NULL;
   }

   if (num_procs > 1)
   {
      void *request;
      hypre_CSRMatrix *Abar, *RbarT, *Pext, *Pbar, *R_diagT, *R_offdT, *Cbar, *Cint, *Cext;
      HYPRE_Int num_cols_offd, local_nnz_Cbar;
      HYPRE_BigInt *col_map_offd;

      hypre_ParCSRMatrixExtractBExtDeviceInit(P, A, 1, &request);

      Abar = hypre_ConcatDiagAndOffdDevice(A);

      if (hypre_ParCSRMatrixDiagT(R))
      {
         R_diagT = hypre_ParCSRMatrixDiagT(R);
      }
      else
      {
         hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      }

      if (hypre_ParCSRMatrixOffdT(R))
      {
         R_offdT = hypre_ParCSRMatrixOffdT(R);
      }
      else
      {
         hypre_CSRMatrixTransposeDevice(R_offd, &R_offdT, 1);
      }

      RbarT = hypre_CSRMatrixStack2Device(R_diagT, R_offdT);

      if (!hypre_ParCSRMatrixDiagT(R))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixDiagT(R) = R_diagT;
         }
         else
         {
            hypre_CSRMatrixDestroy(R_diagT);
         }
      }

      if (!hypre_ParCSRMatrixOffdT(R))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixOffdT(R) = R_offdT;
         }
         else
         {
            hypre_CSRMatrixDestroy(R_offdT);
         }
      }

      Pext = hypre_ParCSRMatrixExtractBExtDeviceWait(request);
      hypre_ConcatDiagOffdAndExtDevice(P, Pext, &Pbar, &num_cols_offd, &col_map_offd);
      hypre_CSRMatrixDestroy(Pext);

      Cbar = hypre_CSRMatrixTripleMultiplyDevice(RbarT, Abar, Pbar);

      hypre_CSRMatrixDestroy(RbarT);
      hypre_CSRMatrixDestroy(Abar);
      hypre_CSRMatrixDestroy(Pbar);

      hypre_assert(hypre_CSRMatrixNumRows(Cbar) ==
                   hypre_ParCSRMatrixNumCols(R) + hypre_CSRMatrixNumCols(R_offd));
      hypre_assert(hypre_CSRMatrixNumCols(Cbar) ==
                   hypre_ParCSRMatrixNumCols(P) + num_cols_offd);

      hypre_TMemcpy(&local_nnz_Cbar,
                    hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(R),
                    HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      // Cint is the bottom part of Cbar
      Cint = hypre_CSRMatrixCreate(hypre_CSRMatrixNumCols(R_offd), hypre_CSRMatrixNumCols(Cbar),
                                   hypre_CSRMatrixNumNonzeros(Cbar) - local_nnz_Cbar);
      hypre_CSRMatrixMemoryLocation(Cint) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixOwnsData(Cint) = 0;

      hypre_CSRMatrixI(Cint) = hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(R);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixI(Cint),
                         hypre_CSRMatrixI(Cint) + hypre_CSRMatrixNumRows(Cint) + 1,
                         hypre_CSRMatrixI(Cint),
      [const_val = local_nnz_Cbar] (const auto & x) {return x - const_val;} );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixI(Cint),
                         hypre_CSRMatrixI(Cint) + hypre_CSRMatrixNumRows(Cint) + 1,
                         thrust::make_constant_iterator(local_nnz_Cbar),
                         hypre_CSRMatrixI(Cint),
                         thrust::minus<HYPRE_Int>() );
#endif

      // Change Cint into a BigJ matrix
      // RL: TODO FIX the 'big' num of columns to global size
      hypre_CSRMatrixBigJ(Cint) = hypre_TAlloc(HYPRE_BigInt,
                                               hypre_CSRMatrixNumNonzeros(Cint),
                                               HYPRE_MEMORY_DEVICE);

      RAP_functor<1, HYPRE_BigInt> func1(hypre_ParCSRMatrixNumCols(P),
                                         hypre_ParCSRMatrixFirstColDiag(P),
                                         col_map_offd);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      hypre_CSRMatrixData(Cint) = hypre_CSRMatrixData(Cbar) + local_nnz_Cbar;

      hypre_ExchangeExternalRowsDeviceInit(Cint, hypre_ParCSRMatrixCommPkg(R), 1, &request);
      Cext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_TFree(hypre_CSRMatrixBigJ(Cint), HYPRE_MEMORY_DEVICE);
      hypre_TFree(Cint, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(R),
                    &local_nnz_Cbar, HYPRE_Int, 1,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

      /* add Cext to local part of Cbar */
      hypre_ParCSRTMatMatPartialAddDevice(hypre_ParCSRMatrixCommPkg(R),
                                          hypre_ParCSRMatrixNumCols(R),
                                          hypre_ParCSRMatrixNumCols(P),
                                          hypre_ParCSRMatrixFirstColDiag(P),
                                          hypre_ParCSRMatrixLastColDiag(P),
                                          num_cols_offd,
                                          col_map_offd,
                                          local_nnz_Cbar,
                                          Cbar,
                                          Cext,
                                          &C_diag,
                                          &C_offd,
                                          &num_cols_offd_C,
                                          &col_map_offd_C);

      hypre_TFree(col_map_offd, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypre_CSRMatrix *R_diagT;
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);

      /* Recover or compute transpose of R_diag */
      if (hypre_ParCSRMatrixDiagT(R))
      {
         R_diagT = hypre_ParCSRMatrixDiagT(R);
      }
      else
      {
         hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      }

      C_diag = hypre_CSRMatrixTripleMultiplyDevice(R_diagT, A_diag, P_diag);
      C_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(R), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);

      /* Keep or destroy transpose of R_diag */
      if (!hypre_ParCSRMatrixDiagT(R))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixDiagT(R) = R_diagT;
         }
         else
         {
            hypre_CSRMatrixDestroy(R_diagT);
         }
      }
   }

   C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumCols(R),
                                hypre_ParCSRMatrixGlobalNumCols(P),
                                hypre_ParCSRMatrixColStarts(R),
                                hypre_ParCSRMatrixColStarts(P),
                                num_cols_offd_C,
                                hypre_CSRMatrixNumNonzeros(C_diag),
                                hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

   hypre_ParCSRMatrixCompressOffdMapDevice(C);
   hypre_ParCSRMatrixCopyColMapOffdToHost(C);

   /* Ensure that the diagonal entries exist in the matrix structure (even if numerically zero) */
   if (hypre_CSRMatrixCheckForMissingDiagonal(C_diag))
   {
      hypre_CSRMatrix *zero = hypre_CSRMatrixIdentityDevice(hypre_CSRMatrixNumRows(C_diag), 0.0);

      hypre_CSRMatrix *C_diag_new = hypre_CSRMatrixAddDevice(1.0, C_diag, 1.0, zero);

      hypre_CSRMatrixDestroy(C_diag);
      hypre_CSRMatrixDestroy(zero);

      hypre_ParCSRMatrixDiag(C) = C_diag_new;
   }

   /* Move the diagonal entry to the first of each row */
   hypre_CSRMatrixMoveDiagFirstDevice(hypre_ParCSRMatrixDiag(C));

   hypre_assert(!hypre_CSRMatrixCheckDiagFirstDevice(hypre_ParCSRMatrixDiag(C)));

   hypre_SyncComputeStream(hypre_handle());

   return C;
}

HYPRE_Int
hypre_ParCSRTMatMatPartialAddDevice( hypre_ParCSRCommPkg *comm_pkg,
                                     HYPRE_Int            num_rows,
                                     HYPRE_Int            num_cols,
                                     HYPRE_BigInt         first_col_diag,
                                     HYPRE_BigInt         last_col_diag,
                                     HYPRE_Int            num_cols_offd,
                                     HYPRE_BigInt        *col_map_offd,
                                     HYPRE_Int            local_nnz_Cbar,
                                     hypre_CSRMatrix     *Cbar,
                                     hypre_CSRMatrix     *Cext,
                                     hypre_CSRMatrix    **C_diag_ptr,
                                     hypre_CSRMatrix    **C_offd_ptr,
                                     HYPRE_Int           *num_cols_offd_C_ptr,
                                     HYPRE_BigInt       **col_map_offd_C_ptr )
{
#if PARCSRGEMM_TIMING > 1
   MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);
   HYPRE_Real t1, t2;
   t1 = hypre_MPI_Wtime();
#endif

   HYPRE_Int        Cext_nnz = hypre_CSRMatrixNumNonzeros(Cext);
   HYPRE_Int        num_cols_offd_C;
   HYPRE_BigInt    *col_map_offd_C;
   hypre_CSRMatrix *Cz;

   // local part of Cbar
   hypre_CSRMatrix *Cbar_local = hypre_CSRMatrixCreate(num_rows, hypre_CSRMatrixNumCols(Cbar),
                                                       local_nnz_Cbar);
   hypre_CSRMatrixI(Cbar_local) = hypre_CSRMatrixI(Cbar);
   hypre_CSRMatrixJ(Cbar_local) = hypre_CSRMatrixJ(Cbar);
   hypre_CSRMatrixData(Cbar_local) = hypre_CSRMatrixData(Cbar);
   hypre_CSRMatrixOwnsData(Cbar_local) = 0;
   hypre_CSRMatrixMemoryLocation(Cbar_local) = HYPRE_MEMORY_DEVICE;

   if (!Cext_nnz)
   {
      num_cols_offd_C = num_cols_offd;
      col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_C, col_map_offd, HYPRE_BigInt, num_cols_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      Cz = Cbar_local;
   }
   else
   {
      in_range<HYPRE_BigInt> pred1(first_col_diag, last_col_diag);

      if (!hypre_CSRMatrixJ(Cext))
      {
         hypre_CSRMatrixJ(Cext) = hypre_TAlloc(HYPRE_Int, Cext_nnz, HYPRE_MEMORY_DEVICE);
      }

      HYPRE_BigInt *Cext_bigj = hypre_CSRMatrixBigJ(Cext);
      HYPRE_BigInt *big_work  = hypre_TAlloc(HYPRE_BigInt, Cext_nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_Int    *work      = hypre_TAlloc(HYPRE_Int, Cext_nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_Int    *map_offd_to_C;

      // Convert Cext from BigJ to J
      // Cext offd
#if defined(HYPRE_USING_SYCL)
      auto off_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj),
                                        oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj) + Cext_nnz,
                                        Cext_bigj,
                                        oneapi::dpl::make_zip_iterator(work, big_work),
                                        std::not_fn(pred1) );

      HYPRE_Int Cext_offd_nnz = std::get<0>(off_end.base()) - work;
#else
      auto off_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), Cext_bigj)),
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                                     Cext_bigj)) + Cext_nnz,
                                        Cext_bigj,
                                        thrust::make_zip_iterator(thrust::make_tuple(work, big_work)),
                                        thrust::not1(pred1) );

      HYPRE_Int Cext_offd_nnz = thrust::get<0>(off_end.get_iterator_tuple()) - work;
#endif

      hypre_CSRMatrixMergeColMapOffd(num_cols_offd, col_map_offd, Cext_offd_nnz, big_work,
                                     &num_cols_offd_C, &col_map_offd_C, &map_offd_to_C);

#if defined(HYPRE_USING_SYCL)
      /* WM: onedpl lower_bound currently does not accept zero length values */
      if (Cext_offd_nnz > 0)
      {
         HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            col_map_offd_C,
                            col_map_offd_C + num_cols_offd_C,
                            big_work,
                            big_work + Cext_offd_nnz,
                            oneapi::dpl::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work) );
      }

      HYPRE_ONEDPL_CALL( std::transform,
                         oneapi::dpl::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
                         oneapi::dpl::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work) + Cext_offd_nnz,
                         oneapi::dpl::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
      [const_val = num_cols] (const auto & x) {return x + const_val;} );
#else
      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         big_work,
                         big_work + Cext_offd_nnz,
                         thrust::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work) );

      HYPRE_THRUST_CALL( transform,
                         thrust::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
                         thrust::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work) + Cext_offd_nnz,
                         thrust::make_constant_iterator(num_cols),
                         thrust::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
                         thrust::plus<HYPRE_Int>() );
#endif

      // Cext diag
#if defined(HYPRE_USING_SYCL)
      auto dia_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj),
                                        oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj) + Cext_nnz,
                                        Cext_bigj,
                                        oneapi::dpl::make_zip_iterator(work, big_work),
                                        pred1 );

      HYPRE_Int Cext_diag_nnz = std::get<0>(dia_end.base()) - work;
#else
      auto dia_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), Cext_bigj)),
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                                     Cext_bigj)) + Cext_nnz,
                                        Cext_bigj,
                                        thrust::make_zip_iterator(thrust::make_tuple(work, big_work)),
                                        pred1 );

      HYPRE_Int Cext_diag_nnz = thrust::get<0>(dia_end.get_iterator_tuple()) - work;
#endif

      hypre_assert(Cext_diag_nnz + Cext_offd_nnz == Cext_nnz);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         big_work,
                         big_work + Cext_diag_nnz,
                         oneapi::dpl::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
      [const_val = first_col_diag](const auto & x) {return x - const_val;} );
#else
      HYPRE_THRUST_CALL( transform,
                         big_work,
                         big_work + Cext_diag_nnz,
                         thrust::make_constant_iterator(first_col_diag),
                         thrust::make_permutation_iterator(hypre_CSRMatrixJ(Cext), work),
                         thrust::minus<HYPRE_BigInt>());
#endif

      hypre_CSRMatrixNumCols(Cext) = num_cols + num_cols_offd_C;

      // transform Cbar_local J index
      RAP_functor<2, HYPRE_Int> func2(num_cols, 0, map_offd_to_C);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixJ(Cbar_local),
                         hypre_CSRMatrixJ(Cbar_local) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar_local),
                         func2 );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(Cbar_local),
                         hypre_CSRMatrixJ(Cbar_local) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar_local),
                         func2 );
#endif

      hypre_CSRMatrixNumCols(Cbar_local) = num_cols + num_cols_offd_C;

      hypre_TFree(big_work,      HYPRE_MEMORY_DEVICE);
      hypre_TFree(work,          HYPRE_MEMORY_DEVICE);
      hypre_TFree(map_offd_to_C, HYPRE_MEMORY_DEVICE);
      hypre_TFree(Cext_bigj,     HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixBigJ(Cext) = NULL;

#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time PartialAdd1 %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif

      // IE = [I, E]
      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

      HYPRE_Int  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int  num_elemt = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      HYPRE_Int *send_map  = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);

      hypre_CSRMatrix *IE = hypre_CSRMatrixCreate(num_rows, num_rows + num_elemt,
                                                  num_rows + num_elemt);
      hypre_CSRMatrixMemoryLocation(IE) = HYPRE_MEMORY_DEVICE;

      HYPRE_Int     *ie_ii = hypre_TAlloc(HYPRE_Int, num_rows + num_elemt, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *ie_j  = hypre_TAlloc(HYPRE_Int, num_rows + num_elemt, HYPRE_MEMORY_DEVICE);
      HYPRE_Complex *ie_a  = NULL;

      if (hypre_HandleSpgemmUseVendor(hypre_handle()))
      {
         ie_a = hypre_TAlloc(HYPRE_Complex, num_rows + num_elemt, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL(std::fill, ie_a, ie_a + num_rows + num_elemt, 1.0);
#else
         HYPRE_THRUST_CALL(fill, ie_a, ie_a + num_rows + num_elemt, 1.0);
#endif
      }

#if defined(HYPRE_USING_SYCL)
      hypreSycl_sequence(ie_ii, ie_ii + num_rows, 0);
      HYPRE_ONEDPL_CALL( std::copy, send_map, send_map + num_elemt, ie_ii + num_rows);
      hypreSycl_sequence(ie_j, ie_j + num_rows + num_elemt, 0);
      auto zipped_begin = oneapi::dpl::make_zip_iterator(ie_ii, ie_j);
      HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + num_rows + num_elemt,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );
#else
      HYPRE_THRUST_CALL( sequence, ie_ii, ie_ii + num_rows);
      HYPRE_THRUST_CALL( copy, send_map, send_map + num_elemt, ie_ii + num_rows);
      HYPRE_THRUST_CALL( sequence, ie_j, ie_j + num_rows + num_elemt);
      HYPRE_THRUST_CALL( stable_sort_by_key, ie_ii, ie_ii + num_rows + num_elemt, ie_j );
#endif

      HYPRE_Int *ie_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, num_rows + num_elemt, ie_ii);
      hypre_TFree(ie_ii, HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrixI(IE)    = ie_i;
      hypre_CSRMatrixJ(IE)    = ie_j;
      hypre_CSRMatrixData(IE) = ie_a;

      // CC = [Cbar_local; Cext]
      hypre_CSRMatrix *CC = hypre_CSRMatrixStack2Device(Cbar_local, Cext);
      hypre_CSRMatrixDestroy(Cbar);
      hypre_CSRMatrixDestroy(Cext);

#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time PartialAdd2 %f\n", t2);
#endif

      // Cz = IE * CC
#if PARCSRGEMM_TIMING > 1
      t1 = hypre_MPI_Wtime();
#endif
      Cz = hypre_CSRMatrixMultiplyDevice(IE, CC);

      hypre_CSRMatrixDestroy(IE);
      hypre_CSRMatrixDestroy(CC);

#if PARCSRGEMM_TIMING > 1
      hypre_ForceSyncComputeStream(hypre_handle());
      t2 = hypre_MPI_Wtime() - t1;
      hypre_ParPrintf(comm, "Time PartialAdd-SpGemm %f\n", t2);
#endif
   }

#if PARCSRGEMM_TIMING > 1
   t1 = hypre_MPI_Wtime();
#endif

   // split into diag and offd
   HYPRE_Int local_nnz_C = hypre_CSRMatrixNumNonzeros(Cz);

   HYPRE_Int     *zmp_i = hypreDevice_CsrRowPtrsToIndices(num_rows, local_nnz_C, hypre_CSRMatrixI(Cz));
   HYPRE_Int     *zmp_j = hypre_CSRMatrixJ(Cz);
   HYPRE_Complex *zmp_a = hypre_CSRMatrixData(Cz);

   in_range<HYPRE_Int> pred(0, num_cols - 1);

#if defined(HYPRE_USING_SYCL)
   HYPRE_Int nnz_C_diag = HYPRE_ONEDPL_CALL( std::count_if,
                                             zmp_j,
                                             zmp_j + local_nnz_C,
                                             pred );
#else
   HYPRE_Int nnz_C_diag = HYPRE_THRUST_CALL( count_if,
                                             zmp_j,
                                             zmp_j + local_nnz_C,
                                             pred );
#endif
   HYPRE_Int nnz_C_offd = local_nnz_C - nnz_C_diag;

   // diag
   hypre_CSRMatrix *C_diag = hypre_CSRMatrixCreate(num_rows, num_cols, nnz_C_diag);
   hypre_CSRMatrixInitialize_v2(C_diag, 0, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_diag_ii = hypre_TAlloc(HYPRE_Int, nnz_C_diag, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_diag_j = hypre_CSRMatrixJ(C_diag);
   HYPRE_Complex *C_diag_a = hypre_CSRMatrixData(C_diag);

#if defined(HYPRE_USING_SYCL)
   auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a),
                                     oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a) + local_nnz_C,
                                     zmp_j,
                                     oneapi::dpl::make_zip_iterator(C_diag_ii, C_diag_j, C_diag_a),
                                     pred );
   hypre_assert( std::get<0>(new_end.base()) == C_diag_ii + nnz_C_diag );
#else
   auto new_end = HYPRE_THRUST_CALL( copy_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)),
                                     thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)) + local_nnz_C,
                                     zmp_j,
                                     thrust::make_zip_iterator(thrust::make_tuple(C_diag_ii, C_diag_j, C_diag_a)),
                                     pred );
   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_diag_ii + nnz_C_diag );
#endif
   hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(C_diag), nnz_C_diag, C_diag_ii,
                                      hypre_CSRMatrixI(C_diag));
   hypre_TFree(C_diag_ii, HYPRE_MEMORY_DEVICE);

   // offd
   hypre_CSRMatrix *C_offd = hypre_CSRMatrixCreate(num_rows, num_cols_offd_C, nnz_C_offd);
   hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_offd_ii = hypre_TAlloc(HYPRE_Int, nnz_C_offd, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_offd_j = hypre_CSRMatrixJ(C_offd);
   HYPRE_Complex *C_offd_a = hypre_CSRMatrixData(C_offd);
#if defined(HYPRE_USING_SYCL)
   new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a),
                                oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a) + local_nnz_C,
                                zmp_j,
                                oneapi::dpl::make_zip_iterator(C_offd_ii, C_offd_j, C_offd_a),
                                std::not_fn(pred) );
   hypre_assert( std::get<0>(new_end.base()) == C_offd_ii + nnz_C_offd );
#else
   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)) + local_nnz_C,
                                zmp_j,
                                thrust::make_zip_iterator(thrust::make_tuple(C_offd_ii, C_offd_j, C_offd_a)),
                                thrust::not1(pred) );
   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_offd_ii + nnz_C_offd );
#endif
   hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(C_offd), nnz_C_offd, C_offd_ii,
                                      hypre_CSRMatrixI(C_offd));
   hypre_TFree(C_offd_ii, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      C_offd_j,
                      C_offd_j + nnz_C_offd,
                      C_offd_j,
   [const_val = num_cols] (const auto & x) {return x - const_val;} );
#else
   HYPRE_THRUST_CALL( transform,
                      C_offd_j,
                      C_offd_j + nnz_C_offd,
                      thrust::make_constant_iterator(num_cols),
                      C_offd_j,
                      thrust::minus<HYPRE_Int>() );
#endif

   // free
   hypre_TFree(Cbar_local, HYPRE_MEMORY_HOST);
   hypre_TFree(zmp_i, HYPRE_MEMORY_DEVICE);

   if (!Cext_nnz)
   {
      hypre_CSRMatrixDestroy(Cbar);
      hypre_CSRMatrixDestroy(Cext);
   }
   else
   {
      hypre_CSRMatrixDestroy(Cz);
   }

#if PARCSRGEMM_TIMING > 1
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_ParPrintf(comm, "Time Split %f\n", t2);
#endif

   // output
   *C_diag_ptr = C_diag;
   *C_offd_ptr = C_offd;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr = col_map_offd_C;

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_GPU) */

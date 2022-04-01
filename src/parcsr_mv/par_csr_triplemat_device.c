/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#define PARCSRGEMM_TIMING 0

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

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
      hypre_ParPrintf(comm, "Size Bext %d %d %d\n", hypre_CSRMatrixNumRows(Bext), hypre_CSRMatrixNumCols(Bext), hypre_CSRMatrixNumNonzeros(Bext));
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
      /* WM: necessary? */
      HYPRE_Int nnz_C_diag = 0;
      if (hypre_CSRMatrixNumNonzeros(Cbar) > 0)
      {
         nnz_C_diag = HYPRE_ONEDPL_CALL( std::count_if,
                                         hypre_CSRMatrixJ(Cbar),
                                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                                         pred );
      }
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
      /* WM: necessary? */
      if (nnz_C_offd > 0)
      {
         HYPRE_ONEDPL_CALL( std::transform,
                            C_offd_j,
                            C_offd_j + nnz_C_offd,
                            C_offd_j,
         [const_val = hypre_ParCSRMatrixNumCols(B)] (const auto & x) {return x - const_val;} );
      }
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

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

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
      hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
      hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
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

      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(A) = AT_diag;
         hypre_ParCSRMatrixOffdT(A) = AT_offd;
      }
      else
      {
         hypre_CSRMatrixDestroy(AT_diag);
         hypre_CSRMatrixDestroy(AT_offd);
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

      hypre_CSRMatrixBigJ(Cint) = hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumNonzeros(Cint),
                                               HYPRE_MEMORY_DEVICE);

      RAP_functor<1, HYPRE_BigInt> func1( hypre_ParCSRMatrixNumCols(B),
                                          hypre_ParCSRMatrixFirstColDiag(B),
                                          hypre_ParCSRMatrixDeviceColMapOffd(B) );
#if defined(HYPRE_USING_SYCL)
      /* WM: necessary? */
      if (hypre_CSRMatrixNumNonzeros(Cbar) > local_nnz_Cbar)
      {
         HYPRE_ONEDPL_CALL( std::transform,
                            hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                            hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                            hypre_CSRMatrixBigJ(Cint),
                            func1 );
      }
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      hypre_ForceSyncComputeStream(hypre_handle());
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
      hypre_ParPrintf(comm, "Size Cext %d %d %d\n", hypre_CSRMatrixNumRows(Cext), hypre_CSRMatrixNumCols(Cext), hypre_CSRMatrixNumNonzeros(Cext));
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
      /* Move the diagonal entry to the first of each row */
      hypre_CSRMatrixMoveDiagFirstDevice(C_diag);
   }

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

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

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
   hypre_CSRMatrix *R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrix *R_offd = hypre_ParCSRMatrixOffd(R);

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix    *C_diag;
   hypre_CSRMatrix    *C_offd;
   HYPRE_Int           num_cols_offd_C = 0;
   HYPRE_BigInt       *col_map_offd_C = NULL;

   HYPRE_Int num_procs;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
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

      hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      hypre_CSRMatrixTransposeDevice(R_offd, &R_offdT, 1);
      RbarT = hypre_CSRMatrixStack2Device(R_diagT, R_offdT);

      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(R) = R_diagT;
         hypre_ParCSRMatrixOffdT(R) = R_offdT;
      }
      else
      {
         hypre_CSRMatrixDestroy(R_diagT);
         hypre_CSRMatrixDestroy(R_offdT);
      }

      Pext = hypre_ParCSRMatrixExtractBExtDeviceWait(request);
      hypre_ConcatDiagOffdAndExtDevice(P, Pext, &Pbar, &num_cols_offd, &col_map_offd);
      hypre_CSRMatrixDestroy(Pext);

      Cbar = hypre_CSRMatrixTripleMultiplyDevice(RbarT, Abar, Pbar);

      hypre_CSRMatrixDestroy(RbarT);
      hypre_CSRMatrixDestroy(Abar);
      hypre_CSRMatrixDestroy(Pbar);

      hypre_assert(hypre_CSRMatrixNumRows(Cbar) == hypre_ParCSRMatrixNumCols(R) + hypre_CSRMatrixNumCols(
                      R_offd));
      hypre_assert(hypre_CSRMatrixNumCols(Cbar) == hypre_ParCSRMatrixNumCols(P) + num_cols_offd);

      hypre_TMemcpy(&local_nnz_Cbar, hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(R), HYPRE_Int, 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

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

      hypre_CSRMatrixBigJ(Cint) = hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumNonzeros(Cint),
                                               HYPRE_MEMORY_DEVICE);

      RAP_functor<1, HYPRE_BigInt> func1(hypre_ParCSRMatrixNumCols(P), hypre_ParCSRMatrixFirstColDiag(P),
                                         col_map_offd);
#if defined(HYPRE_USING_SYCL)
      /* WM: necessary? */
      if (hypre_CSRMatrixNumNonzeros(Cbar) > local_nnz_Cbar)
      {
         HYPRE_ONEDPL_CALL( std::transform,
                            hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                            hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                            hypre_CSRMatrixBigJ(Cint),
                            func1 );
      }
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         hypre_CSRMatrixJ(Cbar) + hypre_CSRMatrixNumNonzeros(Cbar),
                         hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      hypre_ForceSyncComputeStream(hypre_handle());
#endif

      hypre_CSRMatrixData(Cint) = hypre_CSRMatrixData(Cbar) + local_nnz_Cbar;

      hypre_ExchangeExternalRowsDeviceInit(Cint, hypre_ParCSRMatrixCommPkg(R), 1, &request);
      Cext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_TFree(hypre_CSRMatrixBigJ(Cint), HYPRE_MEMORY_DEVICE);
      hypre_TFree(Cint, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_CSRMatrixI(Cbar) + hypre_ParCSRMatrixNumCols(R), &local_nnz_Cbar, HYPRE_Int, 1,
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
      hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      C_diag = hypre_CSRMatrixTripleMultiplyDevice(R_diagT, A_diag, P_diag);
      C_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(R), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(R) = R_diagT;
      }
      else
      {
         hypre_CSRMatrixDestroy(R_diagT);
      }
      /* Move the diagonal entry to the first of each row */
      hypre_CSRMatrixMoveDiagFirstDevice(C_diag);
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

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

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
   t1 = hypre_MPI_Wtime();
#endif
   // to hold Cbar local and Cext
   HYPRE_Int        tmp_s = local_nnz_Cbar + hypre_CSRMatrixNumNonzeros(Cext);
   HYPRE_Int       *tmp_i = hypre_TAlloc(HYPRE_Int,     tmp_s, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *tmp_j = hypre_TAlloc(HYPRE_Int,     tmp_s, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex   *tmp_a = hypre_TAlloc(HYPRE_Complex, tmp_s, HYPRE_MEMORY_DEVICE);
   HYPRE_Int        Cext_diag_nnz, Cext_offd_nnz, num_cols_offd_C, *offd_map_to_C;
   HYPRE_BigInt    *col_map_offd_C;
   hypre_CSRMatrix *C_diag, *C_offd;

   hypre_CSRMatrixSplitDevice_core(0,
                                   hypre_CSRMatrixNumRows(Cext),
                                   hypre_CSRMatrixNumNonzeros(Cext),
                                   NULL,
                                   hypre_CSRMatrixBigJ(Cext), NULL, NULL,
                                   first_col_diag,
                                   last_col_diag,
                                   -1,
                                   NULL, NULL, NULL, NULL,
                                   &Cext_diag_nnz,
                                   NULL, NULL, NULL, NULL,
                                   &Cext_offd_nnz,
                                   NULL, NULL, NULL, NULL);

   HYPRE_Int *Cext_ii = hypreDevice_CsrRowPtrsToIndices(hypre_CSRMatrixNumRows(Cext),
                                                        hypre_CSRMatrixNumNonzeros(Cext),
                                                        hypre_CSRMatrixI(Cext));

   hypre_CSRMatrixSplitDevice_core(1,
                                   hypre_CSRMatrixNumRows(Cext),
                                   hypre_CSRMatrixNumNonzeros(Cext),
                                   Cext_ii,
                                   hypre_CSRMatrixBigJ(Cext),
                                   hypre_CSRMatrixData(Cext),
                                   NULL,
                                   first_col_diag,
                                   last_col_diag,
                                   num_cols_offd,
                                   col_map_offd,
                                   &offd_map_to_C,
                                   &num_cols_offd_C,
                                   &col_map_offd_C,
                                   &Cext_diag_nnz,
                                   tmp_i + local_nnz_Cbar,
                                   tmp_j + local_nnz_Cbar,
                                   tmp_a + local_nnz_Cbar,
                                   NULL,
                                   &Cext_offd_nnz,
                                   tmp_i + local_nnz_Cbar + Cext_diag_nnz,
                                   tmp_j + local_nnz_Cbar + Cext_diag_nnz,
                                   tmp_a + local_nnz_Cbar + Cext_diag_nnz,
                                   NULL);

   hypre_CSRMatrixDestroy(Cext);
   hypre_TFree(Cext_ii, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( tmp_i + local_nnz_Cbar,
                     tmp_i + tmp_s,
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     tmp_i + local_nnz_Cbar );

   /* WM: necessary? */
   if (tmp_s > local_nnz_Cbar + Cext_diag_nnz)
   {
      HYPRE_ONEDPL_CALL( std::transform,
                         tmp_j + local_nnz_Cbar + Cext_diag_nnz,
                         tmp_j + tmp_s,
                         tmp_j + local_nnz_Cbar + Cext_diag_nnz,
                         [const_val = num_cols] (const auto & x) {return x + const_val;} );
   }
#else
   HYPRE_THRUST_CALL( gather,
                      tmp_i + local_nnz_Cbar,
                      tmp_i + tmp_s,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      tmp_i + local_nnz_Cbar );

   HYPRE_THRUST_CALL( transform,
                      tmp_j + local_nnz_Cbar + Cext_diag_nnz,
                      tmp_j + tmp_s,
                      thrust::make_constant_iterator(num_cols),
                      tmp_j + local_nnz_Cbar + Cext_diag_nnz,
                      thrust::plus<HYPRE_Int>() );
#endif

   hypreDevice_CsrRowPtrsToIndices_v2(num_rows, local_nnz_Cbar, hypre_CSRMatrixI(Cbar), tmp_i);
   hypre_TMemcpy(tmp_a, hypre_CSRMatrixData(Cbar), HYPRE_Complex, local_nnz_Cbar, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   RAP_functor<2, HYPRE_Int> func2(num_cols, 0, offd_map_to_C);
#if defined(HYPRE_USING_SYCL)
   /* WM: necessary? */
   if (local_nnz_Cbar > 0)
   {
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixJ(Cbar),
                         hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         tmp_j,
                         func2 );
   }
#else
   HYPRE_THRUST_CALL( transform,
                      hypre_CSRMatrixJ(Cbar),
                      hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                      tmp_j,
                      func2 );
#endif

   hypre_CSRMatrixDestroy(Cbar);
   hypre_TFree(offd_map_to_C, HYPRE_MEMORY_DEVICE);

#if PARCSRGEMM_TIMING > 1
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_ParPrintf(comm, "Time PartialAdd1 %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
   t1 = hypre_MPI_Wtime();
#endif
   // add Cext to Cbar local. Note: type 2, diagonal entries are put at the first in the rows
   hypreDevice_StableSortByTupleKey(tmp_s, tmp_i, tmp_j, tmp_a, 2);

   HYPRE_Int     *zmp_i = hypre_TAlloc(HYPRE_Int,     tmp_s, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *zmp_j = hypre_TAlloc(HYPRE_Int,     tmp_s, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *zmp_a = hypre_TAlloc(HYPRE_Complex, tmp_s, HYPRE_MEMORY_DEVICE);

   HYPRE_Int local_nnz_C = hypreDevice_ReduceByTupleKey(tmp_s, tmp_i, tmp_j, tmp_a, zmp_i, zmp_j, zmp_a);

   hypre_TFree(tmp_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp_a, HYPRE_MEMORY_DEVICE);
#if PARCSRGEMM_TIMING > 1
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_ParPrintf(comm, "Time PartialAdd2 %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
   t1 = hypre_MPI_Wtime();
#endif
   // split into diag and offd
   in_range<HYPRE_Int> pred(0, num_cols - 1);

#if defined(HYPRE_USING_SYCL)
   /* WM: necessary? */
   HYPRE_Int nnz_C_diag = 0;
   if (local_nnz_C > 0)
   {
      nnz_C_diag = HYPRE_ONEDPL_CALL( std::count_if,
                                      zmp_j,
                                      zmp_j + local_nnz_C,
                                      pred );
   }
#else
   HYPRE_Int nnz_C_diag = HYPRE_THRUST_CALL( count_if,
                                             zmp_j,
                                             zmp_j + local_nnz_C,
                                             pred );
#endif
   HYPRE_Int nnz_C_offd = local_nnz_C - nnz_C_diag;

   C_diag = hypre_CSRMatrixCreate(num_rows, num_cols, nnz_C_diag);
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

   C_offd = hypre_CSRMatrixCreate(num_rows, num_cols_offd_C, nnz_C_offd);
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
   /* WM: necessary? */
   if (nnz_C_offd > 0)
   {
      HYPRE_ONEDPL_CALL( std::transform,
                         C_offd_j,
                         C_offd_j + nnz_C_offd,
                         C_offd_j,
                         [const_val = num_cols] (const auto & x) {return x - const_val;} );
   }
#else
   HYPRE_THRUST_CALL( transform,
                      C_offd_j,
                      C_offd_j + nnz_C_offd,
                      thrust::make_constant_iterator(num_cols),
                      C_offd_j,
                      thrust::minus<HYPRE_Int>() );
#endif

   hypre_TFree(zmp_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(zmp_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(zmp_a, HYPRE_MEMORY_DEVICE);
#if PARCSRGEMM_TIMING > 1
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_ParPrintf(comm, "Time Split %f\n", t2);
#endif

   *C_diag_ptr = C_diag;
   *C_offd_ptr = C_offd;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr = col_map_offd_C;

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

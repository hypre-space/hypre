/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_IndepSetMain(hypre_DeviceItem &item,
                            HYPRE_Int   graph_diag_size,
                            HYPRE_Int  *graph_diag,
                            HYPRE_Real *measure_diag,
                            HYPRE_Real *measure_offd,
                            HYPRE_Int  *S_diag_i,
                            HYPRE_Int  *S_diag_j,
                            HYPRE_Int  *S_offd_i,
                            HYPRE_Int  *S_offd_j,
                            HYPRE_Int  *IS_marker_diag,
                            HYPRE_Int  *IS_marker_offd,
                            HYPRE_Int   IS_offd_temp_mark)
{
   HYPRE_Int warp_id = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (warp_id >= graph_diag_size)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int row, row_start, row_end;
   HYPRE_Int i = 0, j;
   HYPRE_Real t = 0.0, measure_row;
   HYPRE_Int marker_row = 1;

   if (lane < 2)
   {
      row = read_only_load(graph_diag + warp_id);
      i   = read_only_load(S_diag_i + row + lane);
   }

   row_start = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 0);
   row_end   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 1);

   if (lane == 0)
   {
      t = read_only_load(measure_diag + row);
   }

   measure_row = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, t, 0);

   for (i = row_start + lane; i < row_end; i += HYPRE_WARP_SIZE)
   {
      j = read_only_load(S_diag_j + i);
      t = read_only_load(measure_diag + j);
      if (t > 1.0)
      {
         if (measure_row > t)
         {
            IS_marker_diag[j] = 0;
         }
         else if (t > measure_row)
         {
            marker_row = 0;
         }
      }
   }

   if (lane < 2)
   {
      i = read_only_load(S_offd_i + row + lane);
   }

   row_start = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 0);
   row_end   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 1);

   for (i = row_start + lane; i < row_end; i += HYPRE_WARP_SIZE)
   {
      j = read_only_load(S_offd_j + i);
      t = read_only_load(measure_offd + j);
      if (t > 1.0)
      {
         if (measure_row > t)
         {
            IS_marker_offd[j] = IS_offd_temp_mark;
         }
         else if (t > measure_row)
         {
            marker_row = 0;
         }
      }
   }

   marker_row = warp_reduce_min(item, marker_row);

   if (lane == 0 && marker_row == 0)
   {
      IS_marker_diag[row] = 0;
   }
}

__global__ void
hypreGPUKernel_IndepSetFixMarker(hypre_DeviceItem &item,
                                 HYPRE_Int  *IS_marker_diag,
                                 HYPRE_Int   num_elmts_send,
                                 HYPRE_Int  *send_map_elmts,
                                 HYPRE_Int  *int_send_buf,
                                 HYPRE_Int   IS_offd_temp_mark)
{
   HYPRE_Int thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (thread_id >= num_elmts_send)
   {
      return;
   }

   if (int_send_buf[thread_id] == IS_offd_temp_mark)
   {
      IS_marker_diag[send_map_elmts[thread_id]] = 0;
   }
}

/* Find IS in the graph whose vertices are in graph_diag, on exit
 * mark the vertices in IS by 1 and those not in IS by 0 in IS_marker_diag
 * Note: IS_marker_offd will not be sync'ed on exit */
HYPRE_Int
hypre_BoomerAMGIndepSetDevice( hypre_ParCSRMatrix  *S,
                               HYPRE_Real          *measure_diag,
                               HYPRE_Real          *measure_offd,
                               HYPRE_Int            graph_diag_size,
                               HYPRE_Int           *graph_diag,
                               HYPRE_Int           *IS_marker_diag,
                               HYPRE_Int           *IS_marker_offd,
                               hypre_ParCSRCommPkg *comm_pkg,
                               HYPRE_Int           *int_send_buf )
{
   /* This a temporary mark used in PMIS alg. to mark the *offd* nodes that
    * should not be in the final IS
    * Must make sure that this number does NOT exist in IS_marker_offd on input
    */
   HYPRE_Int IS_offd_temp_mark = 9999;

   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int  num_sends      = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int  num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);

   hypre_ParCSRCommHandle *comm_handle;

   /*------------------------------------------------------------------
    * Initialize IS_marker by putting all nodes in the IS (marked by 1)
    *------------------------------------------------------------------*/
   hypreDevice_ScatterConstant(IS_marker_diag, graph_diag_size, graph_diag, (HYPRE_Int) 1);

   /*-------------------------------------------------------
    * Remove nodes from the initial independent set
    *-------------------------------------------------------*/
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(graph_diag_size, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_IndepSetMain, gDim, bDim,
                     graph_diag_size, graph_diag, measure_diag, measure_offd,
                     S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                     IS_marker_diag, IS_marker_offd, IS_offd_temp_mark );

   /*--------------------------------------------------------------------
    * Exchange boundary data for IS_marker: send external IS to internal
    *-------------------------------------------------------------------*/
   /* RL: make sure IS_marker_offd is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }

   comm_handle = hypre_ParCSRCommHandleCreate_v2(12, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, IS_marker_offd,
                                                 HYPRE_MEMORY_DEVICE, int_send_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust IS_marker_diag from the received */
   gDim = hypre_GetDefaultDeviceGridDimension(num_elmts_send, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_IndepSetFixMarker, gDim, bDim,
                     IS_marker_diag, num_elmts_send, send_map_elmts,
                     int_send_buf, IS_offd_temp_mark );

   /* Note that IS_marker_offd is not sync'ed (communicated) here */

   return hypre_error_flag;
}

/* Augments measures by some random value between 0 and 1
 * aug_rand: 1: GPU RAND; 11: GPU SEQ RAND
 *           2: CPU RAND; 12: CPU SEQ RAND
 */
HYPRE_Int
hypre_BoomerAMGIndepSetInitDevice( hypre_ParCSRMatrix *S,
                                   HYPRE_Real         *measure_array,
                                   HYPRE_Int           aug_rand)
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(S);
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int        num_rows_diag = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int        my_id;
   HYPRE_Real      *urand;

   hypre_MPI_Comm_rank(comm, &my_id);

   urand = hypre_TAlloc(HYPRE_Real, num_rows_diag, HYPRE_MEMORY_DEVICE);

   if (aug_rand == 2 || aug_rand == 12)
   {
      HYPRE_Real *h_urand;
      h_urand = hypre_CTAlloc(HYPRE_Real, num_rows_diag, HYPRE_MEMORY_HOST);
      hypre_BoomerAMGIndepSetInit(S, h_urand, aug_rand == 12);
      hypre_TMemcpy(urand, h_urand, HYPRE_Real, num_rows_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(h_urand, HYPRE_MEMORY_HOST);
   }
   else if (aug_rand == 11)
   {
      HYPRE_BigInt n_global     = hypre_ParCSRMatrixGlobalNumRows(S);
      HYPRE_BigInt n_first      = hypre_ParCSRMatrixFirstRowIndex(S);
      HYPRE_Real  *urand_global = hypre_TAlloc(HYPRE_Real, n_global, HYPRE_MEMORY_DEVICE);
      // To make sure all rank generate the same sequence
      hypre_CurandUniform(n_global, urand_global, 0, 0, 1, 0);
      hypre_TMemcpy(urand, urand_global + n_first, HYPRE_Real, num_rows_diag, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TFree(urand_global, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypre_assert(aug_rand == 1);
      hypre_CurandUniform(num_rows_diag, urand, 0, 0, 0, 0);
   }

   hypreDevice_ComplexAxpyn(measure_array, num_rows_diag, urand, measure_array, 1.0);

   hypre_TFree(urand, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

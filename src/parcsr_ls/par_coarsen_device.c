/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

#if defined(HYPRE_USING_GPU)

HYPRE_Int hypre_PMISCoarseningInitDevice( hypre_ParCSRMatrix *S, hypre_ParCSRCommPkg *comm_pkg,
                                          HYPRE_Int CF_init, HYPRE_Real *measure_diag, HYPRE_Real *measure_offd, HYPRE_Real *real_send_buf,
                                          HYPRE_Int *graph_diag_size, HYPRE_Int *graph_diag, HYPRE_Int *CF_marker_diag);

HYPRE_Int hypre_PMISCoarseningUpdateCFDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_diag,
                                              HYPRE_Real *measure_offd, HYPRE_Int graph_diag_size, HYPRE_Int *graph_diag,
                                              HYPRE_Int *CF_marker_diag, HYPRE_Int *CF_marker_offd, hypre_ParCSRCommPkg *comm_pkg,
                                              HYPRE_Real *real_send_buf, HYPRE_Int *int_send_buf);

HYPRE_Int
hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix    *S,
                                  hypre_ParCSRMatrix    *A,
                                  HYPRE_Int              CF_init,
                                  HYPRE_Int              debug_flag,
                                  hypre_IntArray       **CF_marker_ptr )
{
   MPI_Comm                  comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;
   hypre_CSRMatrix          *S_diag          = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix          *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                 num_cols_diag   = hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int                 num_cols_offd   = hypre_CSRMatrixNumCols(S_offd);
   HYPRE_Real               *measure_diag;
   HYPRE_Real               *measure_offd;
   HYPRE_Int                 graph_diag_size;
   HYPRE_Int                *graph_diag;
   HYPRE_Int                *diag_iwork;
   HYPRE_Int                *CF_marker_diag;
   HYPRE_Int                *CF_marker_offd;
   HYPRE_Int                 iter = 0;
   void                     *send_buf;
   HYPRE_Int                 my_id, num_procs;
   HYPRE_Int                 aug_rand;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] -= hypre_MPI_Wtime();
#endif

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   /* CF marker */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_cols_diag);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker_diag = hypre_IntArrayData(*CF_marker_ptr);
   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /* arrays for global measure diag and offd parts */
   measure_diag = hypre_TAlloc(HYPRE_Real, num_cols_diag, HYPRE_MEMORY_DEVICE);
   measure_offd = hypre_TAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /* arrays for nodes that are still in the graph (undetermined nodes) */
   graph_diag = hypre_TAlloc(HYPRE_Int, num_cols_diag, HYPRE_MEMORY_DEVICE);

   diag_iwork = hypre_TAlloc(HYPRE_Int, num_cols_diag, HYPRE_MEMORY_DEVICE);

   if ( sizeof(HYPRE_Real) >= sizeof(HYPRE_Int) )
   {
      send_buf = (void *) hypre_TAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                       HYPRE_MEMORY_DEVICE);

   }
   else
   {
      send_buf = (void *) hypre_TAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                       HYPRE_MEMORY_DEVICE);
   }

   /*-------------------------------------------------------------------
    * Compute the global measures
    * The measures are currently given by the column sums of S
    * Hence, measure_array[i] is the number of influences of variable i
    * The measures are augmented by a random number between 0 and 1
    * Note that measure_offd is not sync'ed
    *-------------------------------------------------------------------*/

   if (CF_init == 2 || CF_init == 4)
   {
      /* seq rand */
      aug_rand = hypre_HandleUseGpuRand(hypre_handle()) ? 11 : 12;
   }
   else
   {
      /* each proc generate rand numbers independently */
      aug_rand = hypre_HandleUseGpuRand(hypre_handle()) ? 1 : 2;
   }

   hypre_GetGlobalMeasureDevice(S, comm_pkg, CF_init, aug_rand, measure_diag, measure_offd,
                                (HYPRE_Real *) send_buf);

   /* initialize CF marker, graph arrays and measure_diag, measure_offd is sync'ed
    * Note: CF_marker_offd is not sync'ed */
   hypre_PMISCoarseningInitDevice(S, comm_pkg, CF_init, measure_diag, measure_offd,
                                  (HYPRE_Real *) send_buf,
                                  &graph_diag_size, graph_diag, CF_marker_diag);

   while (1)
   {
      HYPRE_BigInt big_graph_size, global_graph_size;

      big_graph_size = graph_diag_size;

      /* stop the coarsening if nothing left to be coarsened */
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* if (my_id == 0) { hypre_printf("graph size %b\n", global_graph_size); } */

      if (global_graph_size == 0)
      {
         break;
      }

      if (CF_init == 0 || CF_init == 2 || iter)
      {
         /* on input CF_marker_offd does not need to be sync'ed, (but has minimal requirement on
          * the values, see comments therein), and will NOT be sync'ed on exit */
         hypre_BoomerAMGIndepSetDevice(S, measure_diag, measure_offd, graph_diag_size, graph_diag,
                                       CF_marker_diag, CF_marker_offd, comm_pkg, (HYPRE_Int *) send_buf);

         /* sync CF_marker_offd: so it has correct 1/0 now */
#if defined(HYPRE_USING_SYCL)
         hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                           hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                           hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                           CF_marker_diag,
                           (HYPRE_Int *) send_buf );
#else
         HYPRE_THRUST_CALL( gather,
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                            hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                            CF_marker_diag,
                            (HYPRE_Int *) send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
         /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
         if (hypre_GetGpuAwareMPI())
         {
            hypre_ForceSyncComputeStream(hypre_handle());
         }
#endif

         comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg,
                                                       HYPRE_MEMORY_DEVICE, (HYPRE_Int *) send_buf,
                                                       HYPRE_MEMORY_DEVICE, CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      iter ++;

      /* From the IS, set C/F-pts in CF_marker_diag (for the nodes still in graph) and
       * clear their values in measure_diag. measure_offd is sync'ed afterwards.
       * Note: CF_marker_offd Needs to be sync'ed on entry is NOT sync'ed on exit */
      hypre_PMISCoarseningUpdateCFDevice(S, measure_diag, measure_offd, graph_diag_size, graph_diag,
                                         CF_marker_diag, CF_marker_offd, comm_pkg, (HYPRE_Real *) send_buf,
                                         (HYPRE_Int *)send_buf);

      /* Update graph_diag. Remove the nodes with CF_marker_diag != 0 */
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( graph_diag,
                        graph_diag + graph_diag_size,
                        CF_marker_diag,
                        diag_iwork );

      HYPRE_Int *new_end = hypreSycl_remove_if( graph_diag,
                                                graph_diag + graph_diag_size,
                                                diag_iwork,
      [] (const auto & x) {return x;} );
#else
      HYPRE_THRUST_CALL( gather,
                         graph_diag,
                         graph_diag + graph_diag_size,
                         CF_marker_diag,
                         diag_iwork );

      HYPRE_Int *new_end = HYPRE_THRUST_CALL( remove_if,
                                              graph_diag,
                                              graph_diag + graph_diag_size,
                                              diag_iwork,
                                              thrust::identity<HYPRE_Int>() );
#endif

      graph_diag_size = new_end - graph_diag;
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_diag,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(measure_offd,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(graph_diag,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(diag_iwork,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_DEVICE);
   hypre_TFree(send_buf,       HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_GetGlobalMeasureDevice( hypre_ParCSRMatrix  *S,
                              hypre_ParCSRCommPkg *comm_pkg,
                              HYPRE_Int            CF_init,
                              HYPRE_Int            aug_rand,
                              HYPRE_Real          *measure_diag,
                              HYPRE_Real          *measure_offd,
                              HYPRE_Real          *real_send_buf )
{
   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                 num_sends       = hypre_ParCSRCommPkgNumSends(comm_pkg);
   hypre_CSRMatrix          *S_diag          = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix          *S_offd          = hypre_ParCSRMatrixOffd(S);

   /* Compute global column nnz */
   /* compute local column nnz of the offd part */
   hypre_CSRMatrixColNNzRealDevice(S_offd, measure_offd);

   if (hypre_GetGpuAwareMPI())
   {
      /* RL: make sure measure_offd is ready before issuing GPU-GPU MPI */
      hypre_ForceSyncComputeStream(hypre_handle());
   }

   /* send local column nnz of the offd part to neighbors */
   comm_handle = hypre_ParCSRCommHandleCreate_v2(2, comm_pkg, HYPRE_MEMORY_DEVICE, measure_offd,
                                                 HYPRE_MEMORY_DEVICE, real_send_buf);


   /* compute local column nnz of the diag part */
   hypre_CSRMatrixColNNzRealDevice(S_diag, measure_diag);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* add to the local column nnz of the diag part */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   hypreDevice_GenScatterAdd(measure_diag, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                             hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg), real_send_buf, NULL);

   /* Augments the measures with a random number between 0 and 1 (only for the local part) */
   if (aug_rand)
   {
      hypre_BoomerAMGIndepSetInitDevice(S, measure_diag, aug_rand);
   }

   /* Note that measure_offd is not sync'ed (communicated) here
    * and is not set to zero as in the cpu pmis */

   return hypre_error_flag;
}

__global__ void
hypreGPUKernel_PMISCoarseningInit(hypre_DeviceItem &item,
                                  HYPRE_Int   nrows,
                                  HYPRE_Int   CF_init,
                                  HYPRE_Int  *S_diag_i,
                                  HYPRE_Int  *S_offd_i,
                                  HYPRE_Real *measure_diag,
                                  HYPRE_Int  *CF_marker_diag)
{
   /* global_thread_id */
   const HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   HYPRE_Int CF_marker_i = 0;

   if (CF_init == 1)
   {
      // TODO
      hypre_device_assert(0);
   }
   else
   {
      if ( read_only_load(&S_diag_i[i + 1]) - read_only_load(&S_diag_i[i]) == 0 &&
           read_only_load(&S_offd_i[i + 1]) - read_only_load(&S_offd_i[i]) == 0 )
      {
         CF_marker_i = (CF_init == 3 || CF_init == 4) ? C_PT : SF_PT;
         measure_diag[i] = 0.0;
      }
   }

   /*---------------------------------------------
    * If the measure of i is smaller than 1, then
    * make i and F point (because it does not influence
    * any other point)
    * RL: move this step to pmis init and don't do the check
    * in pmis iterations. different from cpu impl
    *---------------------------------------------*/
   if (CF_marker_i == 0 && measure_diag[i] < 1.0)
   {
      CF_marker_i = F_PT;
      measure_diag[i] = 0.0;
   }

   CF_marker_diag[i] = CF_marker_i;
}

HYPRE_Int
hypre_PMISCoarseningInitDevice( hypre_ParCSRMatrix  *S,               /* in */
                                hypre_ParCSRCommPkg *comm_pkg,        /* in */
                                HYPRE_Int            CF_init,         /* in */
                                HYPRE_Real          *measure_diag,    /* in */
                                HYPRE_Real          *measure_offd,    /* out */
                                HYPRE_Real          *real_send_buf,   /* in */
                                HYPRE_Int           *graph_diag_size, /* out */
                                HYPRE_Int           *graph_diag,      /* out */
                                HYPRE_Int           *CF_marker_diag   /* in/out */ )
{
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_diag_i      = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int        num_rows_diag = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int        num_sends     = hypre_ParCSRCommPkgNumSends(comm_pkg);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows_diag, "thread", bDim);

   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *new_end;

   /* init CF_marker_diag and measure_diag: remove some special nodes */
   HYPRE_GPU_LAUNCH( hypreGPUKernel_PMISCoarseningInit, gDim, bDim,
                     num_rows_diag, CF_init, S_diag_i, S_offd_i, measure_diag, CF_marker_diag );

   /* communicate for measure_offd */
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     measure_diag,
                     real_send_buf );
#else
   HYPRE_THRUST_CALL(gather,
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     measure_diag,
                     real_send_buf);
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure real_send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, real_send_buf,
                                                 HYPRE_MEMORY_DEVICE, measure_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* graph_diag consists points with CF_marker_diag == 0 */
#if defined(HYPRE_USING_SYCL)
   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   new_end = hypreSycl_remove_copy_if( count,
                                       count + num_rows_diag,
                                       CF_marker_diag,
                                       graph_diag,
   [] (const auto & x) {return x;} );
#else
   new_end = HYPRE_THRUST_CALL( remove_copy_if,
                                thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(num_rows_diag),
                                CF_marker_diag,
                                graph_diag,
                                thrust::identity<HYPRE_Int>());
#endif

   *graph_diag_size = new_end - graph_diag;

   return hypre_error_flag;
}

__global__ void
hypreGPUKernel_PMISCoarseningUpdateCF(hypre_DeviceItem &item,
                                      HYPRE_Int   graph_diag_size,
                                      HYPRE_Int  *graph_diag,
                                      HYPRE_Int  *S_diag_i,
                                      HYPRE_Int  *S_diag_j,
                                      HYPRE_Int  *S_offd_i,
                                      HYPRE_Int  *S_offd_j,
                                      HYPRE_Real *measure_diag,
                                      HYPRE_Int  *CF_marker_diag,
                                      HYPRE_Int  *CF_marker_offd)
{
   HYPRE_Int warp_id = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (warp_id >= graph_diag_size)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int row = 0, i = 0, marker_row, row_start, row_end;

   if (lane < 2)
   {
      row = read_only_load(graph_diag + warp_id);
      i = read_only_load(CF_marker_diag + row);
   }

   marker_row = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 0);

   if (marker_row > 0)
   {
      if (lane == 0)
      {
         measure_diag[row] = 0.0;
         /* this node is in the IS, mark it as C_PT */
         /* given the fact that C_PT == 1, can skip */
         /*
         CF_marker_diag[row] = C_PT;
         */
      }
   }
   else
   {
      hypre_device_assert(marker_row == 0);

      /*-------------------------------------------------
       * Now treat the case where this node is not in the
       * independent set: loop over
       * all the points j that influence equation 'row'; if
       * any j is a C point, then make row an F point and
       * clear the measure
       *-------------------------------------------------*/
      if (lane < 2)
      {
         i = read_only_load(S_diag_i + row + lane);
      }

      row_start = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 0);
      row_end   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 1);

      for (i = row_start + lane; i < row_end; i += HYPRE_WARP_SIZE)
      {
         HYPRE_Int j = read_only_load(S_diag_j + i);
         /* CF_marker_diag is not r.o. in this kernel */
         HYPRE_Int marker_j = CF_marker_diag[j];

         if (marker_j > 0)
         {
            marker_row = -1;
            break;
         }
      }

      marker_row = warp_allreduce_min(item, marker_row);

      if (marker_row == 0)
      {
         if (lane < 2)
         {
            i = read_only_load(S_offd_i + row + lane);
         }

         row_start = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 0);
         row_end   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, i, 1);

         for (i = row_start + lane; i < row_end; i += HYPRE_WARP_SIZE)
         {
            HYPRE_Int j = read_only_load(S_offd_j + i);
            HYPRE_Int marker_j = read_only_load(CF_marker_offd + j);

            if (marker_j > 0)
            {
               marker_row = -1;
               break;
            }
         }

         marker_row = warp_reduce_min(item, marker_row);
      }

      if (lane == 0 && marker_row == -1)
      {
         CF_marker_diag[row] = F_PT;
         measure_diag[row] = 0.0;
      }
   }
}

HYPRE_Int
hypre_PMISCoarseningUpdateCFDevice( hypre_ParCSRMatrix  *S,               /* in */
                                    HYPRE_Real          *measure_diag,
                                    HYPRE_Real          *measure_offd,
                                    HYPRE_Int            graph_diag_size, /* in */
                                    HYPRE_Int           *graph_diag,      /* in */
                                    HYPRE_Int           *CF_marker_diag,  /* in/out */
                                    HYPRE_Int           *CF_marker_offd,  /* in/out */
                                    hypre_ParCSRCommPkg *comm_pkg,
                                    HYPRE_Real          *real_send_buf,
                                    HYPRE_Int           *int_send_buf )
{
   hypre_CSRMatrix *S_diag    = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i  = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j  = hypre_CSRMatrixJ(S_diag);
   hypre_CSRMatrix *S_offd    = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i  = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j  = hypre_CSRMatrixJ(S_offd);
   HYPRE_Int        num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(graph_diag_size, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_PMISCoarseningUpdateCF,
                     gDim, bDim,
                     graph_diag_size,
                     graph_diag,
                     S_diag_i,
                     S_diag_j,
                     S_offd_i,
                     S_offd_j,
                     measure_diag,
                     CF_marker_diag,
                     CF_marker_offd );

   hypre_ParCSRCommHandle *comm_handle;

   /* communicate for measure_offd */
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     measure_diag,
                     real_send_buf );
#else
   HYPRE_THRUST_CALL(gather,
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     measure_diag,
                     real_send_buf);
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure real_send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, real_send_buf,
                                                 HYPRE_MEMORY_DEVICE, measure_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

#if 0
   /* now communicate CF_marker to CF_marker_offd, to make
      sure that new external F points are known on this processor */
   HYPRE_THRUST_CALL(gather,
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     CF_marker_diag,
                     int_send_buf);

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure int_send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, int_send_buf,
                                                 HYPRE_MEMORY_DEVICE, CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)


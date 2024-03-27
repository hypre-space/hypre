/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

//-----------------------------------------------------------------------
// S_*_j is the special j-array from device SoC
// -1: weak, -2: diag, >=0 (== A_diag_j) : strong
// add weak and the diagonal entries of F-rows
__global__
void hypreGPUKernel_compute_weak_rowsums( hypre_DeviceItem    &item,
                                          HYPRE_Int      nr_of_rows,
                                          bool           has_offd,
                                          HYPRE_Int     *CF_marker,
                                          HYPRE_Int     *A_diag_i,
                                          HYPRE_Complex *A_diag_a,
                                          HYPRE_Int     *Soc_diag_j,
                                          HYPRE_Int     *A_offd_i,
                                          HYPRE_Complex *A_offd_a,
                                          HYPRE_Int     *Soc_offd_j,
                                          HYPRE_Real    *rs,
                                          HYPRE_Int      flag)
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int ib = 0, ie;

   if (lane == 0)
   {
      ib = read_only_load(CF_marker + row);
   }
   ib = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib, 0);

   if (ib >= flag)
   {
      return;
   }

   if (lane < 2)
   {
      ib = read_only_load(A_diag_i + row + lane);
   }
   ie = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib, 1);
   ib = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib, 0);

   HYPRE_Complex rl = 0.0;

   for (HYPRE_Int i = ib + lane; i < ie; i += HYPRE_WARP_SIZE)
   {
      rl += read_only_load(&A_diag_a[i]) * (read_only_load(&Soc_diag_j[i]) < 0);
   }

   if (has_offd)
   {
      if (lane < 2)
      {
         ib = read_only_load(A_offd_i + row + lane);
      }
      ie = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib, 1);
      ib = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib, 0);

      for (HYPRE_Int i = ib + lane; i < ie; i += HYPRE_WARP_SIZE)
      {
         rl += read_only_load(&A_offd_a[i]) * (read_only_load(&Soc_offd_j[i]) < 0);
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rs[row] = rl;
   }
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_compute_aff_afc( hypre_DeviceItem    &item,
                                     HYPRE_Int      nr_of_rows,
                                     HYPRE_Int     *AFF_diag_i,
                                     HYPRE_Int     *AFF_diag_j,
                                     HYPRE_Complex *AFF_diag_data,
                                     HYPRE_Int     *AFF_offd_i,
                                     HYPRE_Complex *AFF_offd_data,
                                     HYPRE_Int     *AFC_diag_i,
                                     HYPRE_Complex *AFC_diag_data,
                                     HYPRE_Int     *AFC_offd_i,
                                     HYPRE_Complex *AFC_offd_data,
                                     HYPRE_Complex *rsW,
                                     HYPRE_Complex *rsFC )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q;

   HYPRE_Complex iscale = 0.0, beta = 0.0;

   if (lane == 0)
   {
      iscale = -1.0 / read_only_load(&rsW[row]);
      beta = read_only_load(&rsFC[row]);
   }
   iscale = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, iscale, 0);
   beta   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, beta,   0);

   // AFF
   /* Diag part */
   if (lane < 2)
   {
      p = read_only_load(AFF_diag_i + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   // do not assume diag is the first element of row
   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         AFF_diag_data[j] = beta * iscale;
      }
      else
      {
         AFF_diag_data[j] *= iscale;
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFF_offd_i + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= iscale;
   }

   if (beta != 0.0)
   {
      beta = 1.0 / beta;
   }

   // AFC
   if (lane < 2)
   {
      p = read_only_load(AFC_diag_i + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   /* Diag part */
   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      AFC_diag_data[j] *= beta;
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFC_offd_i + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      AFC_offd_data[j] *= beta;
   }
}


//-----------------------------------------------------------------------
HYPRE_Int
hypreDevice_extendWtoP( HYPRE_Int      P_nr_of_rows,
                        HYPRE_Int      W_nr_of_rows,
                        HYPRE_Int      W_nr_of_cols,
                        HYPRE_Int     *CF_marker,
                        HYPRE_Int      W_diag_nnz,
                        HYPRE_Int     *W_diag_i,
                        HYPRE_Int     *W_diag_j,
                        HYPRE_Complex *W_diag_data,
                        HYPRE_Int     *P_diag_i,
                        HYPRE_Int     *P_diag_j,
                        HYPRE_Complex *P_diag_data,
                        HYPRE_Int     *W_offd_i,
                        HYPRE_Int     *P_offd_i )
{
   hypre_GpuProfilingPushRange("extendWtoP");

   // row index shift P --> W
   HYPRE_Int *PWoffset = hypre_TAlloc(HYPRE_Int, P_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      CF_marker,
                      &CF_marker[P_nr_of_rows],
                      PWoffset,
                      is_nonnegative<HYPRE_Int>() );
#else
   HYPRE_THRUST_CALL( transform,
                      CF_marker,
                      &CF_marker[P_nr_of_rows],
                      PWoffset,
                      is_nonnegative<HYPRE_Int>() );
#endif

   hypre_Memset(PWoffset + P_nr_of_rows, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   hypreDevice_IntegerExclusiveScan(P_nr_of_rows + 1, PWoffset);

   // map F+C to (next) F
   HYPRE_Int *map2F = hypre_TAlloc(HYPRE_Int, P_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<HYPRE_Int>(P_nr_of_rows + 1),
                      PWoffset,
                      map2F,
                      std::minus<HYPRE_Int>() );
#else
   HYPRE_THRUST_CALL( transform,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(P_nr_of_rows + 1),
                      PWoffset,
                      map2F,
                      thrust::minus<HYPRE_Int>() );
#endif

   // P_diag_i
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( map2F,
                     map2F + P_nr_of_rows + 1,
                     W_diag_i,
                     P_diag_i );

   hypreDevice_IntAxpyn( P_diag_i, P_nr_of_rows + 1, PWoffset, P_diag_i, 1 );

   // P_offd_i
   if (W_offd_i && P_offd_i)
   {
      hypreSycl_gather( map2F,
                        map2F + P_nr_of_rows + 1,
                        W_offd_i,
                        P_offd_i );
   }
#else
   HYPRE_THRUST_CALL( gather,
                      map2F,
                      map2F + P_nr_of_rows + 1,
                      W_diag_i,
                      P_diag_i );

   hypreDevice_IntAxpyn( P_diag_i, P_nr_of_rows + 1, PWoffset, P_diag_i, 1 );

   // P_offd_i
   if (W_offd_i && P_offd_i)
   {
      HYPRE_THRUST_CALL( gather,
                         map2F,
                         map2F + P_nr_of_rows + 1,
                         W_offd_i,
                         P_offd_i );
   }
#endif

   hypre_TFree(map2F, HYPRE_MEMORY_DEVICE);

   // row index shift W --> P
   HYPRE_Int *WPoffset = hypre_TAlloc(HYPRE_Int, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int *new_end = hypreSycl_copy_if( PWoffset,
                                           PWoffset + P_nr_of_rows,
                                           CF_marker,
                                           WPoffset,
                                           is_negative<HYPRE_Int>() );
#else
   HYPRE_Int *new_end = HYPRE_THRUST_CALL( copy_if,
                                           PWoffset,
                                           PWoffset + P_nr_of_rows,
                                           CF_marker,
                                           WPoffset,
                                           is_negative<HYPRE_Int>() );
#endif
   hypre_assert(new_end - WPoffset == W_nr_of_rows);

   hypre_TFree(PWoffset, HYPRE_MEMORY_DEVICE);

   // elements shift
   HYPRE_Int *shift = hypreDevice_CsrRowPtrsToIndices(W_nr_of_rows, W_diag_nnz, W_diag_i);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( shift,
                     shift + W_diag_nnz,
                     WPoffset,
                     shift);
#else
   HYPRE_THRUST_CALL( gather,
                      shift,
                      shift + W_diag_nnz,
                      WPoffset,
                      shift);
#endif

   hypre_TFree(WPoffset, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      shift,
                      shift + W_diag_nnz,
                      oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                      shift,
                      std::plus<HYPRE_Int>() );

   // P_diag_j and P_diag_data
   if (W_diag_j && W_diag_data)
   {
      hypreSycl_scatter( oneapi::dpl::make_zip_iterator(W_diag_j, W_diag_data),
                         oneapi::dpl::make_zip_iterator(W_diag_j, W_diag_data) + W_diag_nnz,
                         shift,
                         oneapi::dpl::make_zip_iterator(P_diag_j, P_diag_data) );
   }
#else
   HYPRE_THRUST_CALL( transform,
                      shift,
                      shift + W_diag_nnz,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      shift,
                      thrust::plus<HYPRE_Int>() );

   // P_diag_j and P_diag_data
   if (W_diag_j && W_diag_data)
   {
      HYPRE_THRUST_CALL( scatter,
                         thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)),
                         thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)) + W_diag_nnz,
                         shift,
                         thrust::make_zip_iterator(thrust::make_tuple(P_diag_j, P_diag_data)) );
   }
#endif
   hypre_TFree(shift, HYPRE_MEMORY_DEVICE);

   // fill the gap
   HYPRE_Int *PC_i = hypre_TAlloc(HYPRE_Int, W_nr_of_cols, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   new_end = hypreSycl_copy_if( P_diag_i,
                                P_diag_i + P_nr_of_rows,
                                CF_marker,
                                PC_i,
                                is_nonnegative<HYPRE_Int>() );
#else
   new_end = HYPRE_THRUST_CALL( copy_if,
                                P_diag_i,
                                P_diag_i + P_nr_of_rows,
                                CF_marker,
                                PC_i,
                                is_nonnegative<HYPRE_Int>() );
#endif

   hypre_assert(new_end - PC_i == W_nr_of_cols);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( copy,
                      oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<HYPRE_Int>(W_nr_of_cols),
                      oneapi::dpl::make_permutation_iterator(P_diag_j, PC_i) );
#else
   HYPRE_THRUST_CALL( scatter,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(W_nr_of_cols),
                      PC_i,
                      P_diag_j );
#endif

   hypreDevice_ScatterConstant(P_diag_data, W_nr_of_cols, PC_i, (HYPRE_Complex) 1.0);

   hypre_TFree(PC_i, HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
// For Ext+i Interp, scale AFF from the left and the right
__global__
void hypreGPUKernel_compute_twiaff_w( hypre_DeviceItem    &item,
                                      HYPRE_Int      nr_of_rows,
                                      HYPRE_BigInt   first_index,
                                      HYPRE_Int     *AFF_diag_i,
                                      HYPRE_Int     *AFF_diag_j,
                                      HYPRE_Complex *AFF_diag_data,
                                      HYPRE_Complex *AFF_diag_data_old,
                                      HYPRE_Int     *AFF_offd_i,
                                      HYPRE_Int     *AFF_offd_j,
                                      HYPRE_Complex *AFF_offd_data,
                                      HYPRE_Int     *AFF_ext_i,
                                      HYPRE_BigInt  *AFF_ext_j,
                                      HYPRE_Complex *AFF_ext_data,
                                      HYPRE_Complex *rsW,
                                      HYPRE_Complex *rsFC,
                                      HYPRE_Complex *rsFC_offd )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);

   HYPRE_Int ib_diag = 0, ie_diag, ib_offd = 0, ie_offd;

   // diag
   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 0);

   HYPRE_Complex theta_i = 0.0;

   // do not assume diag is the first element of row
   // entire warp works on each j
   for (HYPRE_Int indj = ib_diag; indj < ie_diag; indj++)
   {
      HYPRE_Int j = 0;

      if (lane == 0)
      {
         j = read_only_load(&AFF_diag_j[indj]);
      }
      j = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);

      if (j == row)
      {
         if (lane == 0)
         {
            AFF_diag_data[indj] = 1.0;
         }

         continue;
      }

      HYPRE_Int kb = 0, ke;

      // find if there exists entry (j, row) in row j of diag
      if (lane < 2)
      {
         kb = read_only_load(AFF_diag_i + j + lane);
      }
      ke = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, kb, 1);
      kb = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, kb, 0);

      HYPRE_Int kmatch = -1;
      for (HYPRE_Int indk = kb + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, indk < ke);
           indk += HYPRE_WARP_SIZE)
      {
         if (indk < ke && row == read_only_load(&AFF_diag_j[indk]))
         {
            kmatch = indk;
         }

         if (warp_any_sync(item, HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(item, kmatch);

      if (lane == 0)
      {
         HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_diag_data_old[kmatch]) : 0.0;
         HYPRE_Complex rsj = read_only_load(&rsFC[j]) + vji;
         if (rsj)
         {
            HYPRE_Complex vij = read_only_load(&AFF_diag_data_old[indj]) / rsj;
            AFF_diag_data[indj] = vij;
            theta_i += vji * vij;
         }
         else
         {
            AFF_diag_data[indj] = 0.0;
            theta_i += read_only_load(&AFF_diag_data_old[indj]);
         }
      }
   }

   // offd
   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (HYPRE_Int indj = ib_offd; indj < ie_offd; indj++)
   {
      HYPRE_Int j = 0;

      if (lane == 0)
      {
         j = read_only_load(&AFF_offd_j[indj]);
      }
      j = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);

      HYPRE_Int kb = 0, ke;

      if (lane < 2)
      {
         kb = read_only_load(AFF_ext_i + j + lane);
      }
      ke = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, kb, 1);
      kb = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, kb, 0);

      HYPRE_Int kmatch = -1;
      for (HYPRE_Int indk = kb + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, indk < ke);
           indk += HYPRE_WARP_SIZE)
      {
         if (indk < ke && row + first_index == read_only_load(&AFF_ext_j[indk]))
         {
            kmatch = indk;
         }

         if (warp_any_sync(item, HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(item, kmatch);

      if (lane == 0)
      {
         HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_ext_data[kmatch]) : 0.0;
         HYPRE_Complex rsj = read_only_load(&rsFC_offd[j]) + vji;
         if (rsj)
         {
            HYPRE_Complex vij = read_only_load(&AFF_offd_data[indj]) / rsj;
            AFF_offd_data[indj] = vij;
            theta_i += vji * vij;
         }
         else
         {
            AFF_offd_data[indj] = 0.0;
            theta_i += read_only_load(&AFF_offd_data[indj]);
         }
      }
   }

   // scale row
   if (lane == 0)
   {
      theta_i += read_only_load(rsW + row);
      theta_i = theta_i ? -1.0 / theta_i : -1.0;
   }
   theta_i = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, theta_i, 0);

   for (HYPRE_Int j = ib_diag + lane; j < ie_diag; j += HYPRE_WARP_SIZE)
   {
      AFF_diag_data[j] *= theta_i;
   }

   for (HYPRE_Int j = ib_offd + lane; j < ie_offd; j += HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= theta_i;
   }
}


//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_compute_aff_afc_epe( hypre_DeviceItem    &item,
                                         HYPRE_Int      nr_of_rows,
                                         HYPRE_Int     *AFF_diag_i,
                                         HYPRE_Int     *AFF_diag_j,
                                         HYPRE_Complex *AFF_diag_data,
                                         HYPRE_Int     *AFF_offd_i,
                                         HYPRE_Int     *AFF_offd_j,
                                         HYPRE_Complex *AFF_offd_data,
                                         HYPRE_Int     *AFC_diag_i,
                                         HYPRE_Complex *AFC_diag_data,
                                         HYPRE_Int     *AFC_offd_i,
                                         HYPRE_Complex *AFC_offd_data,
                                         HYPRE_Complex *rsW,
                                         HYPRE_Complex *dlam,
                                         HYPRE_Complex *dtmp,
                                         HYPRE_Complex *dtmp_offd )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int pd = 0, qd, po = 0, qo, xd = 0, yd, xo = 0, yo;

   HYPRE_Complex theta = 0.0, value = 0.0;
   HYPRE_Complex dtau_i = 0.0;

   if (lane < 2)
   {
      pd = read_only_load(AFF_diag_i + row + lane);
      po = read_only_load(AFF_offd_i + row + lane);
      xd = read_only_load(AFC_diag_i + row + lane);
      xo = read_only_load(AFC_offd_i + row + lane);
   }

   qd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pd, 1);
   pd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pd, 0);
   qo = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, po, 1);
   po = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, po, 0);
   yd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, xd, 1);
   xd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, xd, 0);
   yo = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, xo, 1);
   xo = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, xo, 0);

   /* D_\tau */
   /* do not assume the first element is the diagonal */
   for (HYPRE_Int j = pd + lane; j < qd; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int index = read_only_load(&AFF_diag_j[j]);
      if (index != row)
      {
         dtau_i += AFF_diag_data[j] * read_only_load(&dtmp[index]);
      }
   }

   for (HYPRE_Int j = po + lane; j < qo; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int index = read_only_load(&AFF_offd_j[j]);
      dtau_i += AFF_offd_data[j] * read_only_load(&dtmp_offd[index]);
   }

   dtau_i = warp_reduce_sum(item, dtau_i);

   if (lane == 0)
   {
      value = read_only_load(&rsW[row]) + dtau_i;
      value = value != 0.0 ? -1.0 / value : 0.0;

      theta = read_only_load(&dlam[row]);
   }

   value = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, value, 0);
   theta = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, theta, 0);

   /* AFF Diag part */
   // do not assume diag is the first element of row
   for (HYPRE_Int j = pd + lane; j < qd; j += HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         AFF_diag_data[j] = theta * value;
      }
      else
      {
         AFF_diag_data[j] *= value;
      }
   }

   /* AFF offd part */
   for (HYPRE_Int j = po + lane; j < qo; j += HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= value;
   }

   theta = theta != 0.0 ? 1.0 / theta : 0.0;

   /* AFC Diag part */
   for (HYPRE_Int j = xd + lane; j < yd; j += HYPRE_WARP_SIZE)
   {
      AFC_diag_data[j] *= theta;
   }

   /* AFC offd part */
   for (HYPRE_Int j = xo + lane; j < yo; j += HYPRE_WARP_SIZE)
   {
      AFC_offd_data[j] *= theta;
   }
}

//-----------------------------------------------------------------------
// For Ext+e Interp, compute D_lambda and D_tmp = D_mu / D_lambda
__global__
void hypreGPUKernel_compute_dlam_dtmp( hypre_DeviceItem    &item,
                                       HYPRE_Int      nr_of_rows,
                                       HYPRE_Int     *AFF_diag_i,
                                       HYPRE_Int     *AFF_diag_j,
                                       HYPRE_Complex *AFF_diag_data,
                                       HYPRE_Int     *AFF_offd_i,
                                       HYPRE_Complex *AFF_offd_data,
                                       HYPRE_Complex *rsFC,
                                       HYPRE_Complex *dlam,
                                       HYPRE_Complex *dtmp )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag = 0, p_offd = 0, q_diag, q_offd;

   if (lane < 2)
   {
      p_diag = read_only_load(AFF_diag_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 0);

   HYPRE_Complex row_sum = 0.0;
   HYPRE_Int find_diag = 0;

   /* do not assume the first element is the diagonal */
   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         find_diag ++;
      }
      else
      {
         row_sum += read_only_load(&AFF_diag_data[j]);
      }
   }

   if (lane < 2)
   {
      p_offd = read_only_load(AFF_offd_i + row + lane);
   }
   q_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      row_sum += read_only_load(&AFF_offd_data[j]);
   }

   row_sum = warp_reduce_sum(item, row_sum);
   find_diag = warp_reduce_sum(item, find_diag);

   if (lane == 0)
   {
      HYPRE_Int num = q_diag - p_diag + q_offd - p_offd - find_diag;
      HYPRE_Complex mu = num > 0 ? row_sum / ((HYPRE_Complex) num) : 0.0;
      /* lambda = beta + mu */
      HYPRE_Complex lam = read_only_load(&rsFC[row]) + mu;
      dlam[row] = lam;
      dtmp[row] = lam != 0.0 ? mu / lam : 0.0;
   }
}

/*---------------------------------------------------------------------
 * Extended Interpolation in the form of Mat-Mat
 *---------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker,
                                    hypre_ParCSRMatrix  *S,
                                    HYPRE_BigInt        *num_cpts_global,
                                    HYPRE_Int            num_functions,
                                    HYPRE_Int           *dof_func,
                                    HYPRE_Int            debug_flag,
                                    HYPRE_Real           trunc_factor,
                                    HYPRE_Int            max_elmts,
                                    hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);

   hypre_ParCSRMatrix *AFF, *AFC;
   hypre_ParCSRMatrix *W, *P;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   HYPRE_Complex      *rsFC, *rsWA, *rsW;
   HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Complex      *P_diag_data;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   /* 0. Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_of_rows, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   hypre_GpuProfilingPushRange("Extract Submatrix");
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   hypre_GpuProfilingPopRange();

   W_nr_of_rows = hypre_ParCSRMatrixNumRows(AFF);
   hypre_assert(A_nr_of_rows == W_nr_of_rows + hypre_ParCSRMatrixNumCols(AFC));

   rsW = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#else
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#endif
   hypre_assert(new_end - rsW == W_nr_of_rows);
   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
   hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_Int *AFF_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Complex *AFF_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Complex *AFF_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Int *AFC_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFC));
   HYPRE_Complex *AFC_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFC));
   HYPRE_Int *AFC_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFC));
   HYPRE_Complex *AFC_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFC));
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_aff_afc,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_a,
                     AFC_diag_i,
                     AFC_diag_a,
                     AFC_offd_i,
                     AFC_offd_a,
                     rsW,
                     rsFC );
   hypre_TFree(rsW,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = hypre_ParCSRMatMatDevice(AFF, AFC);
   hypre_GpuProfilingPopRange();

   hypre_ParCSRMatrixDestroy(AFF);
   hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);

   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(W),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
      hypre_ParCSRMatrixCompressOffdMapDevice(P);
   }
   hypre_GpuProfilingPopRange();

   hypre_MatvecCommPkgCreate(P);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#else
   HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtPIInterpDevice( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker,
                                       hypre_ParCSRMatrix  *S,
                                       HYPRE_BigInt        *num_cpts_global,
                                       HYPRE_Int            num_functions,
                                       HYPRE_Int           *dof_func,
                                       HYPRE_Int            debug_flag,
                                       HYPRE_Real           trunc_factor,
                                       HYPRE_Int            max_elmts,
                                       hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   hypre_CSRMatrix    *AFF_ext = NULL;
   hypre_ParCSRMatrix *AFF, *AFC;
   hypre_ParCSRMatrix *W, *P;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   HYPRE_Complex      *rsFC, *rsFC_offd, *rsWA, *rsW;
   HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i, num_procs;
   HYPRE_Complex      *P_diag_data;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);

   /* 0.Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_of_rows, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp",   bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   hypre_GpuProfilingPushRange("Extract Submatrix");
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   hypre_GpuProfilingPopRange();

   W_nr_of_rows  = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AFF));
   hypre_assert(A_nr_of_rows == W_nr_of_rows + hypre_ParCSRMatrixNumCols(AFC));

   rsW = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#else
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#endif
   hypre_assert(new_end - rsW == W_nr_of_rows);
   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* collect off-processor rsFC */
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(AFF);
      comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   }
   rsFC_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(AFF)),
                            HYPRE_MEMORY_DEVICE);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Complex *send_buf = hypre_TAlloc(HYPRE_Complex, num_elmts_send, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     rsFC,
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      rsFC,
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, rsFC_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   /* offd rows of AFF */
   if (num_procs > 1)
   {
      AFF_ext = hypre_ParCSRMatrixExtractBExtDevice(AFF, AFF, 1);
   }

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   HYPRE_Complex *AFF_diag_data_old = hypre_TAlloc(HYPRE_Complex,
                                                   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(AFF)),
                                                   HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy( AFF_diag_data_old,
                  hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF)),
                  HYPRE_Complex,
                  hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(AFF)),
                  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_BigInt AFF_first_row_idx = hypre_ParCSRMatrixFirstRowIndex(AFF);
   HYPRE_Int *AFF_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Complex *AFF_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Int *AFF_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Complex *AFF_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Int *AFF_ext_i = NULL;
   HYPRE_BigInt *AFF_ext_bigj = NULL;
   HYPRE_Complex *AFF_ext_a = NULL;
   if (AFF_ext)
   {
      AFF_ext_i = hypre_CSRMatrixI(AFF_ext);
      AFF_ext_bigj = hypre_CSRMatrixBigJ(AFF_ext);
      AFF_ext_a = hypre_CSRMatrixData(AFF_ext);
   }
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_twiaff_w,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_first_row_idx,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_diag_data_old,
                     AFF_offd_i,
                     AFF_offd_j,
                     AFF_offd_a,
                     AFF_ext_i,
                     AFF_ext_bigj,
                     AFF_ext_a,
                     rsW,
                     rsFC,
                     rsFC_offd );
   hypre_TFree(rsW,               HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC,              HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC_offd,         HYPRE_MEMORY_DEVICE);
   hypre_TFree(AFF_diag_data_old, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(AFF_ext);
   hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = hypre_ParCSRMatMatDevice(AFF, AFC);
   hypre_GpuProfilingPopRange();

   hypre_ParCSRMatrixDestroy(AFF);
   hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);

   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(W),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
      hypre_ParCSRMatrixCompressOffdMapDevice(P);
   }
   hypre_GpuProfilingPopRange();

   hypre_MatvecCommPkgCreate(P);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#else
   HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

/*---------------------------------------------------------------------
 * Extended+e Interpolation in the form of Mat-Mat
 *---------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtPEInterpDevice(hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      hypre_ParCSRMatrix  *S,
                                      HYPRE_BigInt        *num_cpts_global,
                                      HYPRE_Int            num_functions,
                                      HYPRE_Int           *dof_func,
                                      HYPRE_Int            debug_flag,
                                      HYPRE_Real           trunc_factor,
                                      HYPRE_Int            max_elmts,
                                      hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);
   hypre_ParCSRMatrix *AFF, *AFC;
   hypre_ParCSRMatrix *W, *P;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   HYPRE_Complex      *dlam, *dtmp, *dtmp_offd, *rsFC, *rsWA, *rsW;
   HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Complex      *P_diag_data;

   /* 0. Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_FF) in the notes, only for F-pts */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_of_rows, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   hypre_GpuProfilingPushRange("Extract Submatrix");
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   hypre_GpuProfilingPopRange();

   W_nr_of_rows = hypre_ParCSRMatrixNumRows(AFF);
   hypre_assert(A_nr_of_rows == W_nr_of_rows + hypre_ParCSRMatrixNumCols(AFC));

   rsW = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#else
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
#endif
   hypre_assert(new_end - rsW == W_nr_of_rows);
   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* Generate D_lambda in the paper: D_beta + (row sum of AFF without diagonal elements / row_nnz) */
   /* Generate D_tmp, i.e., D_mu / D_lambda */
   dlam = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   dtmp = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPushRange("Compute D_tmp");
   gDim = hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_Int *AFF_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Complex *AFF_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int *AFF_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Int *AFF_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Complex *AFF_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF));
   HYPRE_Int *AFC_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFC));
   HYPRE_Complex *AFC_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFC));
   HYPRE_Int *AFC_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFC));
   HYPRE_Complex *AFC_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFC));
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_dlam_dtmp,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_a,
                     rsFC,
                     dlam,
                     dtmp );

   /* collect off-processor dtmp */
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(AFF);
      comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   }
   dtmp_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(AFF)),
                            HYPRE_MEMORY_DEVICE);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Complex *send_buf = hypre_TAlloc(HYPRE_Complex, num_elmts_send, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dtmp,
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dtmp,
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, dtmp_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* 4. Form D_tau */
   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
   hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_aff_afc_epe,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_j,
                     AFF_offd_a,
                     AFC_diag_i,
                     AFC_diag_a,
                     AFC_offd_i,
                     AFC_offd_a,
                     rsW,
                     dlam,
                     dtmp,
                     dtmp_offd );
   hypre_TFree(rsW,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC, HYPRE_MEMORY_DEVICE);
   hypre_TFree(dlam, HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp_offd, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = hypre_ParCSRMatMatDevice(AFF, AFC);
   hypre_GpuProfilingPopRange();

   hypre_ParCSRMatrixDestroy(AFF);
   hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);

   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(W),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
      hypre_ParCSRMatrixCompressOffdMapDevice(P);
   }
   hypre_GpuProfilingPopRange();

   hypre_MatvecCommPkgCreate(P);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#else
   HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

#endif // defined(HYPRE_USING_GPU)

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"
#include "struct_matvec_core.h"

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * Initialize StructMatrix/Vector multiplication for variable coeficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_IVC( hypre_StructMatrix *A,
                                    hypre_StructVector *x,
                                    HYPRE_Int           Ab,
                                    HYPRE_Int           depth,
                                    HYPRE_Complex       alpha,
                                    HYPRE_Complex       beta,
                                    HYPRE_Complex      *xp,
                                    HYPRE_Complex      *yp,
                                    HYPRE_Complex      *zp,
                                    HYPRE_Int           ndim,
                                    HYPRE_Int           transpose,
                                    HYPRE_Int           nentries,
                                    HYPRE_Int          *entries,
                                    hypre_Index        *stencil_shape,
                                    hypre_IndexRef      loop_size,
                                    hypre_IndexRef      xfstride,
                                    hypre_IndexRef      start,
                                    hypre_IndexRef      Adstart,
                                    hypre_IndexRef      xdstart,
                                    hypre_IndexRef      ydstart,
                                    hypre_IndexRef      zdstart,
                                    hypre_IndexRef      Adstride,
                                    hypre_IndexRef      xdstride,
                                    hypre_IndexRef      ydstride,
                                    hypre_IndexRef      zdstride,
                                    hypre_Box          *A_data_box,
                                    hypre_Box          *x_data_box,
                                    hypre_Box          *y_data_box,
                                    hypre_Box          *z_data_box)
{
#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   HYPRE_DECLARE_OFFSETS_UP_TO_26;
   HYPRE_DECLARE_OFFSETS(26);
   HYPRE_UNUSED_VAR(nentries);
   HYPRE_UNUSED_VAR(y_data_box);
   HYPRE_UNUSED_VAR(ydstart);
   HYPRE_UNUSED_VAR(ydstride);
   hypre_Index    offset;
   HYPRE_Int      si = 0;

   if (!depth)
   {
      return hypre_error_flag;
   }

#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                               \
   case n:                                                               \
      HYPRE_LOAD_AX_UP_TO_##n(transpose);                                \
      hypre_BoxLoop4Begin(ndim, loop_size,                               \
                          A_data_box, Adstart, Adstride, Ai,             \
                          x_data_box, xdstart, xdstride, xi,             \
                          y_data_box, ydstart, ydstride, yi,             \
                          z_data_box, zdstart, zdstride, zi)             \
      {                                                                  \
         zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_##n); \
      }                                                                  \
      hypre_BoxLoop4End(Ai, xi, yi, zi)                                  \
      break;

   switch (depth)
   {
         HYPRE_CORE_CASE(27);
         HYPRE_CORE_CASE(26);
         HYPRE_CORE_CASE(25);
         HYPRE_CORE_CASE(24);
         HYPRE_CORE_CASE(23);
         HYPRE_CORE_CASE(22);
         HYPRE_CORE_CASE(21);
         HYPRE_CORE_CASE(20);
         HYPRE_CORE_CASE(19);
         HYPRE_CORE_CASE(18);
         HYPRE_CORE_CASE(17);
         HYPRE_CORE_CASE(16);
         HYPRE_CORE_CASE(15);
         HYPRE_CORE_CASE(14);
         HYPRE_CORE_CASE(13);
         HYPRE_CORE_CASE(12);
         HYPRE_CORE_CASE(11);
         HYPRE_CORE_CASE(10);
         HYPRE_CORE_CASE(9);
         HYPRE_CORE_CASE(8);
         HYPRE_CORE_CASE(7);
         HYPRE_CORE_CASE(6);
         HYPRE_CORE_CASE(5);
         HYPRE_CORE_CASE(4);
         HYPRE_CORE_CASE(3);
         HYPRE_CORE_CASE(2);
         HYPRE_CORE_CASE(1);

      case 0:
         break;
   }

   return hypre_error_flag;
}

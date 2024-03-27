/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "red_black_gs.h"

#ifndef hypre_abs
#define hypre_abs(a)  (((a)>0) ? (a) : -(a))
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackConstantCoefGS( void               *relax_vdata,
                              hypre_StructMatrix *A,
                              hypre_StructVector *b,
                              hypre_StructVector *x )
{
   hypre_RedBlackGSData  *relax_data = (hypre_RedBlackGSData  *)relax_vdata;

   HYPRE_Int              max_iter    = (relax_data -> max_iter);
   HYPRE_Int              zero_guess  = (relax_data -> zero_guess);
   HYPRE_Int              rb_start    = (relax_data -> rb_start);
   HYPRE_Int              diag_rank   = (relax_data -> diag_rank);
   hypre_ComputePkg      *compute_pkg = (relax_data -> compute_pkg);
   HYPRE_Int              ndim = hypre_StructMatrixNDim(A);

   hypre_CommHandle      *comm_handle;

   hypre_BoxArrayArray   *compute_box_aa;
   hypre_BoxArray        *compute_box_a;
   hypre_Box             *compute_box;

   hypre_Box             *A_dbox;
   hypre_Box             *b_dbox;
   hypre_Box             *x_dbox;

   HYPRE_Int              Ai, Astart, Ani, Anj;
   HYPRE_Int              bstart, bni, bnj;
   HYPRE_Int              xstart, xni, xnj;
   HYPRE_Int              xoff0 = 0, xoff1 = 0, xoff2 = 0;
   HYPRE_Int              xoff3 = 0, xoff4 = 0, xoff5 = 0;

   HYPRE_Real            *Ap;
   HYPRE_Real            *App;
   HYPRE_Real            *bp;
   HYPRE_Real            *xp;

   /* constant coefficient */
   HYPRE_Int              constant_coeff = hypre_StructMatrixConstantCoefficient(A);
   HYPRE_Real             App0 = 1.0, App1 = 1.0, App2 = 1.0;
   HYPRE_Real             App3 = 1.0, App4 = 1.0, App5 = 1.0, AApd = 1.0;

   hypre_IndexRef         start;
   hypre_Index            loop_size;

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
   HYPRE_Int              offd[6];

   HYPRE_Int              iter, rb, redblack, d;
   HYPRE_Int              compute_i, i, j;
   HYPRE_Int              ni, nj, nk;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   hypre_BeginTiming(relax_data -> time_index);

   hypre_StructMatrixDestroy(relax_data -> A);
   hypre_StructVectorDestroy(relax_data -> b);
   hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> x) = hypre_StructVectorRef(x);
   (relax_data -> b) = hypre_StructVectorRef(b);

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(relax_data -> time_index);
      return hypre_error_flag;
   }
   else
   {
      stencil       = hypre_StructMatrixStencil(A);
      stencil_shape = hypre_StructStencilShape(stencil);
      stencil_size  = hypre_StructStencilSize(stencil);

      /* get off-diag entry ranks ready */
      i = 0;
      for (j = 0; j < stencil_size; j++)
      {
         if (j != diag_rank)
         {
            offd[i] = j;
            i++;
         }
      }
   }

   hypre_StructVectorClearBoundGhostValues(x, 0);

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   rb = rb_start;
   iter = 0;

   if (zero_guess)
   {
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            b_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
            x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);

            Ap = hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = hypre_StructVectorBoxData(b, i);
            xp = hypre_StructVectorBoxData(x, i);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               start  = hypre_BoxIMin(compute_box);
               hypre_BoxGetSize(compute_box, loop_size);

               /* Are we relaxing index start or start+(1,0,0)? */
               redblack = rb;
               for (d = 0; d < ndim; d++)
               {
                  redblack += hypre_IndexD(start, d);
               }
               redblack = hypre_abs(redblack) % 2;

               bstart = hypre_BoxIndexRank(b_dbox, start);
               xstart = hypre_BoxIndexRank(x_dbox, start);
               ni = hypre_IndexX(loop_size);
               nj = hypre_IndexY(loop_size);
               nk = hypre_IndexZ(loop_size);
               bni = hypre_BoxSizeX(b_dbox);
               xni = hypre_BoxSizeX(x_dbox);
               bnj = hypre_BoxSizeY(b_dbox);
               xnj = hypre_BoxSizeY(x_dbox);
               if (ndim < 3)
               {
                  nk = 1;
                  if (ndim < 2)
                  {
                     nj = 1;
                  }
               }

               if (constant_coeff == 1)
               {
                  Ai = hypre_CCBoxIndexRank(A_dbox, start);
                  AApd = 1.0 / Ap[Ai];

                  hypre_RedBlackLoopInit();

#define DEVICE_VAR is_device_ptr(xp,bp)
                  hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                      bstart, bni, bnj, bi,
                                                      xstart, xni, xnj, xi);
                  {
                     xp[xi] = bp[bi] * AApd;
                  }
                  hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
               }

               else      /* variable coefficient diag */
               {
                  Astart = hypre_BoxIndexRank(A_dbox, start);
                  Ani = hypre_BoxSizeX(A_dbox);
                  Anj = hypre_BoxSizeY(A_dbox);

                  hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                  hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                          Astart, Ani, Anj, Ai,
                                          bstart, bni, bnj, bi,
                                          xstart, xni, xnj, xi);
                  {
                     xp[xi] = bp[bi] / Ap[Ai];
                  }
                  hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
               }

            }
         }
      }

      rb = (rb + 1) % 2;
      iter++;
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < 2 * max_iter)
   {
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x);
               hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            b_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
            x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);

            Ap = hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = hypre_StructVectorBoxData(b, i);
            xp = hypre_StructVectorBoxData(x, i);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               start  = hypre_BoxIMin(compute_box);
               hypre_BoxGetSize(compute_box, loop_size);

               /* Are we relaxing index start or start+(1,0,0)? */
               redblack = rb;
               for (d = 0; d < ndim; d++)
               {
                  redblack += hypre_IndexD(start, d);
               }
               redblack = hypre_abs(redblack) % 2;

               bstart = hypre_BoxIndexRank(b_dbox, start);
               xstart = hypre_BoxIndexRank(x_dbox, start);
               ni = hypre_IndexX(loop_size);
               nj = hypre_IndexY(loop_size);
               nk = hypre_IndexZ(loop_size);
               bni = hypre_BoxSizeX(b_dbox);
               xni = hypre_BoxSizeX(x_dbox);
               bnj = hypre_BoxSizeY(b_dbox);
               xnj = hypre_BoxSizeY(x_dbox);
               Ai = hypre_CCBoxIndexRank(A_dbox, start);
               if (ndim < 3)
               {
                  nk = 1;
                  if (ndim < 2)
                  {
                     nj = 1;
                  }
               }

               switch (stencil_size)
               {
                  case 7:
                     App = hypre_StructMatrixBoxData(A, i, offd[5]);
                     App5 = App[Ai];
                     App = hypre_StructMatrixBoxData(A, i, offd[4]);
                     App4 = App[Ai];
                     xoff5 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[5]]);
                     xoff4 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[4]]);
                  /* fall through */

                  case 5:
                     App = hypre_StructMatrixBoxData(A, i, offd[3]);
                     App3 = App[Ai];
                     App = hypre_StructMatrixBoxData(A, i, offd[2]);
                     App2 = App[Ai];
                     xoff3 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[3]]);
                     xoff2 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[2]]);
                  /* fall through */

                  case 3:
                     App = hypre_StructMatrixBoxData(A, i, offd[1]);
                     App1 = App[Ai];
                     App = hypre_StructMatrixBoxData(A, i, offd[0]);
                     App0 = App[Ai];
                     xoff1 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[1]]);
                     xoff0 = hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[0]]);
                     break;
               }

               if (constant_coeff == 1)
               {
                  AApd = 1 / Ap[Ai];

                  switch (stencil_size)
                  {
                     case 7:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3] -
                               App4 * xp[xi + xoff4] -
                               App5 * xp[xi + xoff5]) * AApd;
                        }
                        hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR

                        break;

                     case 5:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3]) * AApd;
                        }
                        hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 3:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1]) * AApd;
                        }
                        hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
                        break;
                  }

               }  /* if (constant_coeff == 1) */

               else /* variable diagonal */
               {
                  Astart = hypre_BoxIndexRank(A_dbox, start);
                  Ani = hypre_BoxSizeX(A_dbox);
                  Anj = hypre_BoxSizeY(A_dbox);

                  switch (stencil_size)
                  {
                     case 7:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3] -
                               App4 * xp[xi + xoff4] -
                               App5 * xp[xi + xoff5]) / Ap[Ai];
                        }
                        hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 5:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3]) / Ap[Ai];
                        }
                        hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 3:
                        hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1]) / Ap[Ai];
                        }
                        hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                  }  /* switch(stencil_size) */
               }     /* else */
            }
         }
      }

      rb = (rb + 1) % 2;
      iter++;
   }

   (relax_data -> num_iterations) = iter / 2;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(relax_data -> flops);
   hypre_EndTiming(relax_data -> time_index);

   return hypre_error_flag;
}

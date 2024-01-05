/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * This routine assumes a 3-pt (1D), 5-pt (2D), or 7-pt (3D) stencil.
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "red_black_gs.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_RedBlackGSCreate( MPI_Comm  comm )
{
   hypre_RedBlackGSData *relax_data;

   relax_data = hypre_CTAlloc(hypre_RedBlackGSData,  1, HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = hypre_InitializeTiming("RedBlackGS");

   /* set defaults */
   (relax_data -> tol)         = 1.0e-06;
   (relax_data -> max_iter)    = 1000;
   (relax_data -> rel_change)  = 0;
   (relax_data -> zero_guess)  = 0;
   (relax_data -> rb_start)    = 1;
   (relax_data -> flops)       = 0;
   (relax_data -> A)           = NULL;
   (relax_data -> b)           = NULL;
   (relax_data -> x)           = NULL;
   (relax_data -> compute_pkg) = NULL;

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSDestroy( void *relax_vdata )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   if (relax_data)
   {
      hypre_StructMatrixDestroy(relax_data -> A);
      hypre_StructVectorDestroy(relax_data -> b);
      hypre_StructVectorDestroy(relax_data -> x);
      hypre_ComputePkgDestroy(relax_data -> compute_pkg);

      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetup( void               *relax_vdata,
                       hypre_StructMatrix *A,
                       hypre_StructVector *b,
                       hypre_StructVector *x )
{
   hypre_RedBlackGSData  *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   HYPRE_Int              diag_rank;
   hypre_ComputePkg      *compute_pkg;

   hypre_StructGrid      *grid;
   hypre_StructStencil   *stencil;
   hypre_Index            diag_index;
   hypre_ComputeInfo     *compute_info;

   /*----------------------------------------------------------
    * Find the matrix diagonal
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   hypre_SetIndex3(diag_index, 0, 0, 0);
   diag_rank = hypre_StructStencilElementRank(stencil, diag_index);

   /*----------------------------------------------------------
    * Set up the compute packages
    *----------------------------------------------------------*/

   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(x), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> x) = hypre_StructVectorRef(x);
   (relax_data -> b) = hypre_StructVectorRef(b);
   (relax_data -> diag_rank) = diag_rank;
   (relax_data -> compute_pkg) = compute_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGS( void               *relax_vdata,
                  hypre_StructMatrix *A,
                  hypre_StructVector *b,
                  hypre_StructVector *x )
{
   hypre_RedBlackGSData  *relax_data = (hypre_RedBlackGSData *)relax_vdata;

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

   HYPRE_Int              Astart, Ani, Anj;
   HYPRE_Int              bstart, bni, bnj;
   HYPRE_Int              xstart, xni, xnj;
   HYPRE_Int              xoff0, xoff1, xoff2, xoff3, xoff4, xoff5;

   HYPRE_Real            *Ap;
   HYPRE_Real            *Ap0, *Ap1, *Ap2, *Ap3, *Ap4, *Ap5;
   HYPRE_Real            *bp;
   HYPRE_Real            *xp;

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

               Astart = hypre_BoxIndexRank(A_dbox, start);
               bstart = hypre_BoxIndexRank(b_dbox, start);
               xstart = hypre_BoxIndexRank(x_dbox, start);
               ni = hypre_IndexX(loop_size);
               nj = hypre_IndexY(loop_size);
               nk = hypre_IndexZ(loop_size);
               Ani = hypre_BoxSizeX(A_dbox);
               bni = hypre_BoxSizeX(b_dbox);
               xni = hypre_BoxSizeX(x_dbox);
               Anj = hypre_BoxSizeY(A_dbox);
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

               Astart = hypre_BoxIndexRank(A_dbox, start);
               bstart = hypre_BoxIndexRank(b_dbox, start);
               xstart = hypre_BoxIndexRank(x_dbox, start);
               ni = hypre_IndexX(loop_size);
               nj = hypre_IndexY(loop_size);
               nk = hypre_IndexZ(loop_size);
               Ani = hypre_BoxSizeX(A_dbox);
               bni = hypre_BoxSizeX(b_dbox);
               xni = hypre_BoxSizeX(x_dbox);
               Anj = hypre_BoxSizeY(A_dbox);
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

               switch (stencil_size)
               {
                  case 7:
                     Ap5 = hypre_StructMatrixBoxData(A, i, offd[5]);
                     Ap4 = hypre_StructMatrixBoxData(A, i, offd[4]);
                     xoff5 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[5]]);
                     xoff4 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[4]]);
                  // fall through

                  case 5:
                     Ap3 = hypre_StructMatrixBoxData(A, i, offd[3]);
                     Ap2 = hypre_StructMatrixBoxData(A, i, offd[2]);
                     xoff3 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[3]]);
                     xoff2 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[2]]);
                  // fall through

                  case 3:
                     Ap1 = hypre_StructMatrixBoxData(A, i, offd[1]);
                     Ap0 = hypre_StructMatrixBoxData(A, i, offd[0]);
                     xoff1 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[1]]);
                     xoff0 = hypre_BoxOffsetDistance(x_dbox, stencil_shape[offd[0]]);
                     break;
               }

               switch (stencil_size)
               {
                  case 7:
                     hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap)
                     hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                             Astart, Ani, Anj, Ai,
                                             bstart, bni, bnj, bi,
                                             xstart, xni, xnj, xi);
                     {
                        xp[xi] =
                           (bp[bi] -
                            Ap0[Ai] * xp[xi + xoff0] -
                            Ap1[Ai] * xp[xi + xoff1] -
                            Ap2[Ai] * xp[xi + xoff2] -
                            Ap3[Ai] * xp[xi + xoff3] -
                            Ap4[Ai] * xp[xi + xoff4] -
                            Ap5[Ai] * xp[xi + xoff5]) / Ap[Ai];
                     }
                     hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                     break;

                  case 5:
                     hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap0,Ap1,Ap2,Ap3,Ap)
                     hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                             Astart, Ani, Anj, Ai,
                                             bstart, bni, bnj, bi,
                                             xstart, xni, xnj, xi);
                     {
                        xp[xi] =
                           (bp[bi] -
                            Ap0[Ai] * xp[xi + xoff0] -
                            Ap1[Ai] * xp[xi + xoff1] -
                            Ap2[Ai] * xp[xi + xoff2] -
                            Ap3[Ai] * xp[xi + xoff3]) / Ap[Ai];
                     }
                     hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                     break;

                  case 3:
                     hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap0,Ap1,Ap)
                     hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                             Astart, Ani, Anj, Ai,
                                             bstart, bni, bnj, bi,
                                             xstart, xni, xnj, xi);
                     {
                        xp[xi] =
                           (bp[bi] -
                            Ap0[Ai] * xp[xi + xoff0] -
                            Ap1[Ai] * xp[xi + xoff1]) / Ap[Ai];
                     }
                     hypre_RedBlackLoopEnd();
#undef DEVICE_VAR

                     break;
               }
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetTol( void   *relax_vdata,
                        HYPRE_Real  tol )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   (relax_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetMaxIter( void *relax_vdata,
                            HYPRE_Int   max_iter )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetZeroGuess( void *relax_vdata,
                              HYPRE_Int   zero_guess )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetStartRed( void *relax_vdata )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   (relax_data -> rb_start) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RedBlackGSSetStartBlack( void *relax_vdata )
{
   hypre_RedBlackGSData *relax_data = (hypre_RedBlackGSData *)relax_vdata;

   (relax_data -> rb_start) = 0;

   return hypre_error_flag;
}

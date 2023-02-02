/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/*--------------------------------------------------------------------------
 * hypre_StructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix  *A;
   hypre_StructVector  *x;
   hypre_ComputePkg    *compute_pkg;

} hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_StructMatvecCreate( void )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData,  1, HYPRE_MEMORY_HOST);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSetup( void               *matvec_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *x            )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_StructGrid        *grid;
   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(x), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = hypre_StructMatrixRef(A);
   (matvec_data -> x)           = hypre_StructVectorRef(x);
   (matvec_data -> compute_pkg) = compute_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute( void               *matvec_vdata,
                           HYPRE_Complex       alpha,
                           hypre_StructMatrix *A,
                           hypre_StructVector *x,
                           HYPRE_Complex       beta,
                           hypre_StructVector *y            )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_ComputePkg        *compute_pkg;

   hypre_CommHandle        *comm_handle;

   hypre_BoxArrayArray     *compute_box_aa;
   hypre_Box               *y_data_box;

   HYPRE_Complex           *xp;
   HYPRE_Complex           *yp;

   hypre_BoxArray          *boxes;
   hypre_Box               *box;
   hypre_Index              loop_size;
   hypre_IndexRef           start;
   hypre_IndexRef           stride;

   HYPRE_Int                constant_coefficient;

   HYPRE_Complex            temp;
   HYPRE_Int                compute_i, i;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) { hypre_StructVectorClearBoundGhostValues(x, 0); }

   compute_pkg = (matvec_data -> compute_pkg);

   stride = hypre_ComputePkgStride(compute_pkg);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
      hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp)
         hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                             y_data_box, start, stride, yi);
         {
            yp[yi] *= beta;
         }
         hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
      }

      return hypre_error_flag;
   }

   if (x == y)
   {
      x_tmp = hypre_StructVectorClone(y);
      x = x_tmp;
   }
   /*-----------------------------------------------------------------------
    * Do (alpha != 0.0) computation
    *-----------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y normally (where everything
             * is multiplied by alpha at the end),
             * beta*y for constant coefficient (where only Ax gets multiplied by alpha)
             *--------------------------------------------------------------*/

            if ( constant_coefficient == 1 )
            {
               temp = beta;
            }
            else
            {
               temp = beta / alpha;
            }
            if (temp != 1.0)
            {
               boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
               hypre_ForBoxI(i, boxes)
               {
                  box   = hypre_BoxArrayBox(boxes, i);
                  start = hypre_BoxIMin(box);

                  y_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  yp = hypre_StructVectorBoxData(y, i);

#define DEVICE_VAR is_device_ptr(yp)
                  if (temp == 0.0)
                  {
                     hypre_BoxGetSize(box, loop_size);

                     hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                                         y_data_box, start, stride, yi);
                     {
                        yp[yi] = 0.0;
                     }
                     hypre_BoxLoop1End(yi);
                  }
                  else
                  {
                     hypre_BoxGetSize(box, loop_size);

                     hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                                         y_data_box, start, stride, yi);
                     {
                        yp[yi] *= temp;
                     }
                     hypre_BoxLoop1End(yi);
                  }
#undef DEVICE_VAR
               }
            }
         }
         break;

         case 1:
         {
            hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * y += A*x
       *--------------------------------------------------------------------*/

      switch ( constant_coefficient )
      {
         case 0:
         {
            hypre_StructMatvecCC0( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
         case 1:
         {
            hypre_StructMatvecCC1( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
         case 2:
         {
            hypre_StructMatvecCC2( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
      }

   }

   if (x_tmp)
   {
      hypre_StructVectorDestroy(x_tmp);
      x = y;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCC0
 * core of struct matvec computation, for the case constant_coefficient==0
 * (all coefficients are variable)
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecCC0( HYPRE_Complex       alpha,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *x,
                                 hypre_StructVector *y,
                                 hypre_BoxArrayArray     *compute_box_aa,
                                 hypre_IndexRef           stride
                               )
{
   HYPRE_Int i, j, si;
   HYPRE_Complex           *Ap0;
   HYPRE_Complex           *Ap1;
   HYPRE_Complex           *Ap2;
   HYPRE_Complex           *Ap3;
   HYPRE_Complex           *Ap4;
   HYPRE_Complex           *Ap5;
   HYPRE_Complex           *Ap6;
   HYPRE_Int                xoff0;
   HYPRE_Int                xoff1;
   HYPRE_Int                xoff2;
   HYPRE_Int                xoff3;
   HYPRE_Int                xoff4;
   HYPRE_Int                xoff5;
   HYPRE_Int                xoff6;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_Box               *A_data_box;
   hypre_Box               *x_data_box;
   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   hypre_Box               *y_data_box;
   HYPRE_Complex           *xp;
   HYPRE_Complex           *yp;
   HYPRE_Int                depth;
   hypre_Index              loop_size;
   hypre_IndexRef           start;
   HYPRE_Int                ndim;

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   ndim          = hypre_StructVectorNDim(x);

   hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, j);

         hypre_BoxGetSize(compute_box, loop_size);
         start  = hypre_BoxIMin(compute_box);

         /* unroll up to depth MAX_DEPTH */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = hypre_StructMatrixBoxData(A, i, si + 6);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4] +
                        Ap5[Ai] * xp[xi + xoff5] +
                        Ap6[Ai] * xp[xi + xoff6];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 6:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4] +
                        Ap5[Ai] * xp[xi + xoff5];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 5:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 4:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 3:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 2:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 1:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;
            }
         }

         if (alpha != 1.0)
         {
#define DEVICE_VAR is_device_ptr(yp)
            hypre_BoxLoop1Begin(ndim, loop_size,
                                y_data_box, start, stride, yi);
            {
               yp[yi] *= alpha;
            }
            hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
         }
      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructMatvecCC1
 * core of struct matvec computation, for the case constant_coefficient==1
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecCC1( HYPRE_Complex       alpha,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *x,
                                 hypre_StructVector *y,
                                 hypre_BoxArrayArray     *compute_box_aa,
                                 hypre_IndexRef           stride
                               )
{
   HYPRE_Int i, j, si;
   HYPRE_Complex           *Ap0;
   HYPRE_Complex           *Ap1;
   HYPRE_Complex           *Ap2;
   HYPRE_Complex           *Ap3;
   HYPRE_Complex           *Ap4;
   HYPRE_Complex           *Ap5;
   HYPRE_Complex           *Ap6;
   HYPRE_Complex           AAp0;
   HYPRE_Complex           AAp1;
   HYPRE_Complex           AAp2;
   HYPRE_Complex           AAp3;
   HYPRE_Complex           AAp4;
   HYPRE_Complex           AAp5;
   HYPRE_Complex           AAp6;
   HYPRE_Int                xoff0;
   HYPRE_Int                xoff1;
   HYPRE_Int                xoff2;
   HYPRE_Int                xoff3;
   HYPRE_Int                xoff4;
   HYPRE_Int                xoff5;
   HYPRE_Int                xoff6;
   HYPRE_Int                Ai;

   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_Box               *x_data_box;
   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   hypre_Box               *y_data_box;
   HYPRE_Complex           *xp;
   HYPRE_Complex           *yp;
   HYPRE_Int                depth;
   hypre_Index              loop_size;
   hypre_IndexRef           start;
   HYPRE_Int                ndim;

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   ndim          = hypre_StructVectorNDim(x);

   hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, j);

         hypre_BoxGetSize(compute_box, loop_size);
         start  = hypre_BoxIMin(compute_box);

         Ai = 0;

         /* unroll up to depth MAX_DEPTH */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = hypre_StructMatrixBoxData(A, i, si + 6);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;
                  AAp5 = Ap5[Ai] * alpha;
                  AAp6 = Ap6[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5] +
                        AAp6 * xp[xi + xoff6];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 6:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;
                  AAp5 = Ap5[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 5:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 4:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 3:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 2:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 1:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  AAp0 = Ap0[Ai] * alpha;

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
            }
         }
      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructMatvecCC2
 * core of struct matvec computation, for the case constant_coefficient==2
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructMatvecCC2( HYPRE_Complex       alpha,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *x,
                                 hypre_StructVector *y,
                                 hypre_BoxArrayArray     *compute_box_aa,
                                 hypre_IndexRef           stride
                               )
{
   HYPRE_Int i, j, si;
   HYPRE_Complex           *Ap0;
   HYPRE_Complex           *Ap1;
   HYPRE_Complex           *Ap2;
   HYPRE_Complex           *Ap3;
   HYPRE_Complex           *Ap4;
   HYPRE_Complex           *Ap5;
   HYPRE_Complex           *Ap6;
   HYPRE_Complex           AAp0;
   HYPRE_Complex           AAp1;
   HYPRE_Complex           AAp2;
   HYPRE_Complex           AAp3;
   HYPRE_Complex           AAp4;
   HYPRE_Complex           AAp5;
   HYPRE_Complex           AAp6;
   HYPRE_Int                xoff0;
   HYPRE_Int                xoff1;
   HYPRE_Int                xoff2;
   HYPRE_Int                xoff3;
   HYPRE_Int                xoff4;
   HYPRE_Int                xoff5;
   HYPRE_Int                xoff6;
   HYPRE_Int                si_center, center_rank;
   hypre_Index              center_index;
   HYPRE_Int                Ai_CC;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_Box               *A_data_box;
   hypre_Box               *x_data_box;
   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   hypre_Box               *y_data_box;
   HYPRE_Complex           *xp;
   HYPRE_Complex           *yp;
   HYPRE_Int                depth;
   hypre_Index              loop_size;
   hypre_IndexRef           start;
   HYPRE_Int                ndim;
   HYPRE_Complex            zero[1] = {0};

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   ndim          = hypre_StructVectorNDim(x);

   hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, j);

         hypre_BoxGetSize(compute_box, loop_size);
         start  = hypre_BoxIMin(compute_box);

         Ai_CC = hypre_CCBoxIndexRank( A_data_box, start );

         /* Find the stencil index for the center of the stencil, which
            makes the matrix diagonal.  This is the variable coefficient
            part of the matrix, so will get different treatment...*/
         hypre_SetIndex(center_index, 0);
         center_rank = hypre_StructStencilElementRank( stencil, center_index );
         si_center = center_rank;

         /* unroll up to depth MAX_DEPTH
            Only the constant coefficient part of the matrix is referenced here,
            the center (variable) coefficient part is deferred. */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = hypre_StructMatrixBoxData(A, i, si + 6);
                  if ( (0 <= si_center - si) && (si_center - si < 7) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                        case 5: Ap5 = zero; break;
                        case 6: Ap6 = zero; break;
                     }
                  }

                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];
                  AAp5 = Ap5[Ai_CC];
                  AAp6 = Ap6[Ai_CC];


                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5] +
                        AAp6 * xp[xi + xoff6];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR

                  break;

               case 6:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = hypre_StructMatrixBoxData(A, i, si + 5);
                  if ( (0 <= si_center - si) && (si_center - si < 6) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                        case 5: Ap5 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];
                  AAp5 = Ap5[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 5:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = hypre_StructMatrixBoxData(A, i, si + 4);
                  if ( (0 <= si_center - si) && (si_center - si < 5) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 4:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = hypre_StructMatrixBoxData(A, i, si + 3);
                  if ( (0 <= si_center - si) && (si_center - si < 4) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 3:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = hypre_StructMatrixBoxData(A, i, si + 2);
                  if ( (0 <= si_center - si) && (si_center - si < 3) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 2:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = hypre_StructMatrixBoxData(A, i, si + 1);
                  if ( (0 <= si_center - si) && (si_center - si < 2) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 1:
                  Ap0 = hypre_StructMatrixBoxData(A, i, si + 0);
                  if ( si_center - si == 0 )
                  {
                     Ap0 = zero;
                  }
                  AAp0 = Ap0[Ai_CC];

                  xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0];
                  }
                  hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR

                  break;
            }
         }

         Ap0 = hypre_StructMatrixBoxData(A, i, si_center);
         xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                         stencil_shape[si_center]);
         if (alpha != 1.0 )
         {
#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                y_data_box, start, stride, yi);
            {
               yp[yi] = alpha * ( yp[yi] +
                                  Ap0[Ai] * xp[xi + xoff0] );
            }
            hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
         }
         else
         {
#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                y_data_box, start, stride, yi);
            {
               yp[yi] +=
                  Ap0[Ai] * xp[xi + xoff0];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
         }

      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
   hypre_StructMatvecData *matvec_data = (hypre_StructMatvecData *)matvec_vdata;

   if (matvec_data)
   {
      hypre_StructMatrixDestroy(matvec_data -> A);
      hypre_StructVectorDestroy(matvec_data -> x);
      hypre_ComputePkgDestroy(matvec_data -> compute_pkg );
      hypre_TFree(matvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvec( HYPRE_Complex       alpha,
                    hypre_StructMatrix *A,
                    hypre_StructVector *x,
                    HYPRE_Complex       beta,
                    hypre_StructVector *y     )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

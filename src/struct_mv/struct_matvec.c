/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 27

/*--------------------------------------------------------------------------
 * Matvec data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix  *A;
   hypre_StructVector  *x;
   hypre_ComputePkg    *compute_pkg;
   hypre_BoxArray      *data_space;
   HYPRE_Int            transpose;

} hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructMatvecCreate( )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData, 1);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSetTranspose( void *matvec_vdata,
                                HYPRE_Int transpose )
{
   hypre_StructMatvecData  *matvec_data = matvec_vdata;

   (matvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSetup( void               *matvec_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *x )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_StructGrid        *grid;
   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;

   hypre_IndexRef           dom_stride;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Make sure that the transpose coefficients are stored in A (note that the
    * resizing will provide no option to restore A to its previous state) */
   if (matvec_data -> transpose)
   {
      HYPRE_Int  resize;

      hypre_StructMatrixSetTranspose(A, 1, &resize);
      if ( (resize) && (hypre_StructMatrixDataSpace(A) != NULL) )
      {
         hypre_BoxArray  *data_space;

         hypre_StructMatrixComputeDataSpace(A, NULL, &data_space);
         hypre_StructMatrixResize(A, data_space);
         hypre_StructMatrixForget(A);  /* No restore allowed (saves memory) */
         hypre_StructMatrixAssemble(A);
      }
   }

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);
   if (matvec_data -> transpose)
   {
      dom_stride = hypre_StructMatrixRanStride(A);
   }
   else
   {
      dom_stride = hypre_StructMatrixDomStride(A);
   }

   /* This computes a data_space with respect to the matrix grid and the
    * stencil pattern for the matvec */
   hypre_StructVectorReindex(x, grid, dom_stride);
   hypre_StructNumGhostFromStencil(stencil, &num_ghost);
   hypre_StructVectorComputeDataSpace(x, num_ghost, &data_space);
   hypre_TFree(num_ghost);

   /* This computes the communication pattern for the new x data_space */
   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_StructVectorMapCommInfo(x, hypre_ComputeInfoCommInfo(compute_info));
   /* Compute boxes will be appropriately projected in MatvecCompute */
   hypre_ComputePkgCreate(compute_info, data_space, 1, grid, &compute_pkg);

   /* This restores the original grid */
   hypre_StructVectorRestore(x);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = hypre_StructMatrixRef(A);
   (matvec_data -> x)           = hypre_StructVectorRef(x);
   (matvec_data -> compute_pkg) = compute_pkg;
   (matvec_data -> data_space)  = data_space;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute( void               *matvec_vdata,
                           HYPRE_Complex       alpha,
                           hypre_StructMatrix *A,
                           hypre_StructVector *x,
                           HYPRE_Complex       beta,
                           hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data   = (hypre_StructMatvecData  *)matvec_vdata;
   HYPRE_Int                transpose     = (matvec_data -> transpose);
   HYPRE_Int                dom_is_coarse = hypre_StructMatrixDomainIsCoarse(A);
   HYPRE_Int                ran_is_coarse = hypre_StructMatrixRangeIsCoarse(A);

   if (transpose || dom_is_coarse || ran_is_coarse)
   {
      hypre_StructMatvecRectglCompute(matvec_vdata, alpha, A, x, beta, y);
   }
   else
   {
      hypre_StructMatvecSquareCompute(matvec_vdata, alpha, A, x, beta, y);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *
 * StructMatvec specialization for rectangular matrices.
 *
 * The grids for A, x, and y must be compatible with respect to matrix-vector
 * multiply, but the initial grid boxes and strides may differ.  The routines
 * Reindex() and Resize() are called to convert the grid for the vector x to
 * match the domain grid of the matrix A.  As a result, both A and x have the
 * same list of underlying boxes and the domain stride for A is the same as the
 * stride for x.  The grid for y is assumed to match the range grid for A, but
 * with potentially different boxes and strides.  The number of boxes and the
 * corresponding box ids must match, however, so it is not necessary to search.
 * Here are some examples (after Reindex() and Resize() have been called for x):
 *
 *   RangeIsCoarse:                           DomainIsCoarse:
 *   Adstride = 1                             Adstride = 1
 *   xdstride = 3                             xdstride = 1
 *   ydstride = 1                             ydstride = 3
 *
 *   1     6               2 2                5     2     6 6    <-- domain/range strides
 *   | |   |               | | |              | |   |     | | |
 *   |y| = |       A       | | |              | |   |     | |x|
 *   | |   |               | |x|              |y| = |  A  | | |
 *                           | |              | |   |     |
 *                           | |              | |   |     |
 *                           | |              | |   |     |
 *
 * It is assumed here that the data space for y corresponds to a coarsening of
 * the base index space for A with range stride.  So, we are circumventing the
 * "MapData" routines in the matrix and vector classes to avoid making too many
 * function calls.  We could achieve the same goal by calling MapToCoarse(),
 * MapToFine(), and MapData() in sequence to move base index values first to the
 * range grid for A, then to the base index space and data space for y.
 *
 * RDF TODO: Consider modifications to the current DataMap interfaces to avoid
 * making assumptions like the above.  Look at the Matmult routine as well.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecRectglCompute( void               *matvec_vdata,
                                 HYPRE_Complex       alpha,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *x,
                                 HYPRE_Complex       beta,
                                 hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_ComputePkg        *compute_pkg;
   hypre_CommHandle        *comm_handle;
   HYPRE_Int                transpose;
   HYPRE_Int                ndim;

   hypre_BoxArray          *data_space;
   hypre_BoxArrayArray     *compute_box_aa;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_BoxArray          *boxes;
   hypre_Index              loop_size, origin, stride;
   hypre_IndexRef           start;

   HYPRE_Complex            temp;
   HYPRE_Int                compute_i, i, j, si;

   hypre_Box               *A_data_box, *x_data_box, *y_data_box;
   HYPRE_Complex           *Ap, *xp, *yp;
   HYPRE_Int                Ai, xi, yi;
   HYPRE_Int 		    Ab, xb, yb;
   hypre_Index              Adstride, xdstride, ydstride, ustride;

   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   hypre_StructGrid        *grid;
   HYPRE_Int 		    ran_nboxes;
   HYPRE_Int 		   *ran_boxnums;
   hypre_IndexRef           ran_stride;
   hypre_IndexRef           dom_stride;
   HYPRE_Int                dom_is_coarse;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

#if 0
   /* RDF: Should not need this if the boundaries were cleared initially */
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) hypre_StructVectorClearBoundGhostValues(x, 0);
#endif

   compute_pkg = (matvec_data -> compute_pkg);
   transpose   = (matvec_data -> transpose);

   ndim = hypre_StructMatrixNDim(A);

   /* Switch domain and range information if doing a transpose matvec */
   if (transpose)
   {
      ran_stride    = hypre_StructMatrixDomStride(A);
      ran_nboxes    = hypre_StructMatrixDomNBoxes(A);
      ran_boxnums   = hypre_StructMatrixDomBoxnums(A);
      dom_stride    = hypre_StructMatrixRanStride(A);
      dom_is_coarse = hypre_StructMatrixRangeIsCoarse(A);
   }
   else
   {
      ran_stride    = hypre_StructMatrixRanStride(A);
      ran_nboxes    = hypre_StructMatrixRanNBoxes(A);
      ran_boxnums   = hypre_StructMatrixRanBoxnums(A);
      dom_stride    = hypre_StructMatrixDomStride(A);
      dom_is_coarse = hypre_StructMatrixDomainIsCoarse(A);
   }

   grid = hypre_StructMatrixGrid(A);

   compute_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
      hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
         hypre_StructVectorMapDataBox(y, compute_box);
         hypre_BoxGetSize(compute_box, loop_size);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         start = hypre_BoxIMin(compute_box);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxLoop1Begin(ndim, loop_size,
                             y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoop1For(yi)
         {
            yp[yi] *= beta;
         }
         hypre_BoxLoop1End(yi);
      }
      HYPRE_ANNOTATE_FUNC_END;

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

   /* This resizes the data for x using the data_space computed during setup */
   data_space = hypre_BoxArrayClone(matvec_data -> data_space);
   hypre_StructVectorReindex(x, grid, dom_stride);
   hypre_StructVectorResize(x, data_space);

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y normally (where everything
             * is multiplied by alpha at the end),
             *--------------------------------------------------------------*/

            temp = beta / alpha;
            if (temp != 1.0)
            {
               boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
               hypre_ForBoxI(i, boxes)
               {
                  hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
                  hypre_StructVectorMapDataBox(y, compute_box);
                  hypre_BoxGetSize(compute_box, loop_size);
                  y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  start = hypre_BoxIMin(compute_box);
                  yp = hypre_StructVectorBoxData(y, i);

                  if (temp == 0.0)
                  {
                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop1For(yi)
                     {
                        yp[yi] = 0.0;
                     }
                     hypre_BoxLoop1End(yi);
                  }
                  else
                  {
                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop1For(yi)
                     {
                        yp[yi] *= temp;
                     }
                     hypre_BoxLoop1End(yi);
                  }
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
       * y += A*x (or A^T*x)
       *--------------------------------------------------------------------*/

      hypre_StructMatrixGetStencilStride(A, stride);
      hypre_CopyToIndex(stride, ndim, Adstride);
      hypre_StructMatrixMapDataStride(A, Adstride);
      hypre_CopyToIndex(stride, ndim, xdstride);
      hypre_StructVectorMapDataStride(x, xdstride);
      hypre_CopyToIndex(stride, ndim, ydstride);
      hypre_MapToCoarseIndex(ydstride, NULL, ran_stride, ndim);

      yb = 0;
      for (i = 0; i < ran_nboxes; i++)
      {
         hypre_Index  Adstart;
         hypre_Index  xdstart;
         hypre_Index  ydstart;

         /* The corresponding box IDs for the following boxnums should match */
         Ab = ran_boxnums[i];
         xb = Ab;  /* Reindex ensures that A and x have the same grid boxes */
         yb = hypre_StructVectorBoxnum(y, i);

         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, Ab);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), Ab);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), xb);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), yb);

         xp = hypre_StructVectorBoxData(x, xb);
         yp = hypre_StructVectorBoxData(y, yb);

         hypre_ForBoxI(j, compute_box_a)
         {
            /* TODO (later, for optimization): Unroll these loops */
            for (si = 0; si < stencil_size; si++)
            {
               /* If the domain grid is coarse, the compute box will change
                * based on the stencil entry.  Otherwise, the next code block
                * needs to be called only on the first stencil iteration.
                *
                * Note that the Adstart and xdstart values are set in different
                * places depending on the value of transpose. */

               if ((si == 0) || dom_is_coarse)
               {
                  hypre_CopyBox(hypre_BoxArrayBox(compute_box_a, j), compute_box);
                  hypre_StructMatrixGetStencilSpace(A, si, transpose, origin, stride);
                  hypre_ProjectBox(compute_box, origin, stride);
                  start = hypre_BoxIMin(compute_box);

                  if (!transpose) /* Set Adstart here and xdstart below */
                  {
                     hypre_CopyToIndex(start, ndim, Adstart);
                     hypre_StructMatrixMapDataIndex(A, Adstart);
                  }

                  hypre_CopyToIndex(start, ndim, ydstart);
                  hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);

                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);
               }

               if (transpose)
               {
                  hypre_SubtractIndexes(start, stencil_shape[si], ndim, Adstart);
                  hypre_StructMatrixMapDataIndex(A, Adstart);
                  hypre_SubtractIndexes(start, stencil_shape[si], ndim, xdstart);
                  hypre_StructVectorMapDataIndex(x, xdstart);
               }
               else /* Set Adstart above and xdstart here */
               {
                  hypre_AddIndexes(start, stencil_shape[si], ndim, xdstart);
                  hypre_StructVectorMapDataIndex(x, xdstart);
               }

               Ap = hypre_StructMatrixBoxData(A, Ab, si);
               if (hypre_StructMatrixConstEntry(A, si))
               {
                  /* Constant coefficient case */
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, xdstart, xdstride, xi,
                                      y_data_box, ydstart, ydstride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop2For(xi, yi)
                  {
                     yp[yi] += Ap[0] * xp[xi];
                  }
                  hypre_BoxLoop2End(xi, yi);
               }
               else
               {
                  /* Variable coefficient case */
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, Adstart, Adstride, Ai,
                                      x_data_box, xdstart, xdstride, xi,
                                      y_data_box, ydstart, ydstride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi,Ai) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap[Ai] * xp[xi];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
               }
            }

            if (alpha != 1.0)
            {
               hypre_BoxLoop1Begin(ndim, loop_size,
                                   y_data_box, ydstart, ydstride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
               hypre_BoxLoop1For(yi)
               {
                  yp[yi] *= alpha;
               }
               hypre_BoxLoop1End(yi);
            }
         }
      }
   }

   if (x_tmp)
   {
      hypre_StructVectorDestroy(x_tmp);
      x = y;
   }
   hypre_BoxDestroy(compute_box);

   /* This restores the original grid and data layout */
   hypre_StructVectorRestore(x);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * StructMatvec specialization for square matrices.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSquareCompute( void               *matvec_vdata,
                                 HYPRE_Complex       alpha,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *x,
                                 HYPRE_Complex       beta,
                                 hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_ComputePkg        *compute_pkg = (matvec_data -> compute_pkg);
   hypre_BoxArray          *data_space  = (matvec_data -> data_space);
   HYPRE_Int                transpose   = (matvec_data -> transpose);
   HYPRE_Int                ndim        = hypre_StructMatrixNDim(A);

   hypre_CommHandle        *comm_handle;
   hypre_BoxArrayArray     *compute_box_aa;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_BoxArray          *boxes;
   hypre_Index              loop_size, origin, ustride;
   hypre_IndexRef           start;

   HYPRE_Complex            temp;
   HYPRE_Int                compute_i, i, j;

   HYPRE_Complex           *Ap,   *xp,   *yp;
   HYPRE_Complex           *Ap0,  *Ap1,  *Ap2;
   HYPRE_Complex           *Ap3,  *Ap4,  *Ap5;
   HYPRE_Complex           *Ap6,  *Ap7,  *Ap8;
   HYPRE_Complex           *Ap9,  *Ap10, *Ap11;
   HYPRE_Complex           *Ap12, *Ap13, *Ap14;
   HYPRE_Complex           *Ap15, *Ap16, *Ap17;
   HYPRE_Complex           *Ap18, *Ap19, *Ap20;
   HYPRE_Complex           *Ap21, *Ap22, *Ap23;
   HYPRE_Complex           *Ap24, *Ap25, *Ap26;

   HYPRE_Int                xoff0,  xoff1,  xoff2;
   HYPRE_Int                xoff3,  xoff4,  xoff5;
   HYPRE_Int                xoff6,  xoff7,  xoff8;
   HYPRE_Int                xoff9,  xoff10, xoff11;
   HYPRE_Int                xoff12, xoff13, xoff14;
   HYPRE_Int                xoff15, xoff16, xoff17;
   HYPRE_Int                xoff18, xoff19, xoff20;
   HYPRE_Int                xoff21, xoff22, xoff23;
   HYPRE_Int                xoff24, xoff25, xoff26;

   hypre_Box               *A_data_box, *x_data_box, *y_data_box;
   HYPRE_Int                si, sk, ssi[MAX_DEPTH], depth, k;
   HYPRE_Int                Ai, xi, yi;

   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Safety checks */
   if (hypre_StructMatrixDomainIsCoarse(A) || hypre_StructMatrixRangeIsCoarse(A))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "A is not square!");
      return hypre_error_flag;
   }

   if (transpose)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Transpose operation not implemeted!");
      return hypre_error_flag;
   }

   hypre_SetIndex(ustride, 1);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/
   if (alpha == 0.0)
   {
      hypre_StructScale(beta, y);

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------------
    * Do (alpha != 0.0) computation
    *-----------------------------------------------------------------------*/
   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   if (x == y)
   {
      x_tmp = hypre_StructVectorClone(y);
      x = x_tmp;
   }

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y normally (where everything
             * is multiplied by alpha at the end),
             *--------------------------------------------------------------*/
            temp = beta / alpha;
            hypre_StructScale(temp, y);
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
       * y += alpha*A*x
       *--------------------------------------------------------------------*/

       hypre_ForBoxArrayI(i, compute_box_aa)
       {
          compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

          A_data_box =  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
          x_data_box =  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
          y_data_box =  hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
          yp = hypre_StructVectorBoxData(y, i);

          hypre_ForBoxArrayI(j, compute_box_a)
          {
            compute_box = hypre_BoxArrayBox(compute_box_a, j);
            start = hypre_BoxIMin(compute_box);
            hypre_BoxGetStrideSize(compute_box, ustride, loop_size);

            /* unroll up to depth MAX_DEPTH */
            for (si = 0; si < stencil_size; si += MAX_DEPTH)
            {
               depth = hypre_min(MAX_DEPTH, (stencil_size - si));

               for (k = 0, sk = si; k < depth; sk++)
               {
                  ssi[k] = sk;
                  k++;
               }

               switch (depth)
               {
                  case 27:
                     Ap26 = hypre_StructMatrixBoxData(A, i, ssi[26]);
                     xoff26 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[26]]);

                  case 26:
                     Ap25 = hypre_StructMatrixBoxData(A, i, ssi[25]);
                     xoff25 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[25]]);

                  case 25:
                     Ap24 = hypre_StructMatrixBoxData(A, i, ssi[24]);
                     xoff24 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[24]]);

                  case 24:
                     Ap23 = hypre_StructMatrixBoxData(A, i, ssi[23]);
                     xoff23 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[23]]);

                  case 23:
                     Ap22 = hypre_StructMatrixBoxData(A, i, ssi[22]);
                     xoff22 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[22]]);

                  case 22:
                     Ap21 = hypre_StructMatrixBoxData(A, i, ssi[21]);
                     xoff21 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[21]]);

                  case 21:
                     Ap20 = hypre_StructMatrixBoxData(A, i, ssi[20]);
                     xoff20 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[20]]);

                  case 20:
                     Ap19 = hypre_StructMatrixBoxData(A, i, ssi[19]);
                     xoff19 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[19]]);

                  case 19:
                     Ap18 = hypre_StructMatrixBoxData(A, i, ssi[18]);
                     xoff18 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[18]]);

                  case 18:
                     Ap17 = hypre_StructMatrixBoxData(A, i, ssi[17]);
                     xoff17 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[17]]);

                  case 17:
                     Ap16 = hypre_StructMatrixBoxData(A, i, ssi[16]);
                     xoff16 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[16]]);

                  case 16:
                     Ap15 = hypre_StructMatrixBoxData(A, i, ssi[15]);
                     xoff15 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[15]]);

                  case 15:
                     Ap14 = hypre_StructMatrixBoxData(A, i, ssi[14]);
                     xoff14 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[14]]);

                  case 14:
                     Ap13 = hypre_StructMatrixBoxData(A, i, ssi[13]);
                     xoff13 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[13]]);

                  case 13:
                     Ap12 = hypre_StructMatrixBoxData(A, i, ssi[12]);
                     xoff12 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[12]]);

                  case 12:
                     Ap11 = hypre_StructMatrixBoxData(A, i, ssi[11]);
                     xoff11 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[11]]);

                  case 11:
                     Ap10 = hypre_StructMatrixBoxData(A, i, ssi[10]);
                     xoff10 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[10]]);

                  case 10:
                     Ap9 = hypre_StructMatrixBoxData(A, i, ssi[9]);
                     xoff9 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[9]]);

                  case 9:
                     Ap8 = hypre_StructMatrixBoxData(A, i, ssi[8]);
                     xoff8 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[8]]);

                  case 8:
                     Ap7 = hypre_StructMatrixBoxData(A, i, ssi[7]);
                     xoff7 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[7]]);

                  case 7:
                     Ap6 = hypre_StructMatrixBoxData(A, i, ssi[6]);
                     xoff6 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[6]]);

                  case 6:
                     Ap5 = hypre_StructMatrixBoxData(A, i, ssi[5]);
                     xoff5 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[5]]);

                  case 5:
                     Ap4 = hypre_StructMatrixBoxData(A, i, ssi[4]);
                     xoff4 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[4]]);

                  case 4:
                     Ap3 = hypre_StructMatrixBoxData(A, i, ssi[3]);
                     xoff3 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[3]]);

                  case 3:
                     Ap2 = hypre_StructMatrixBoxData(A, i, ssi[2]);
                     xoff2 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[2]]);

                  case 2:
                     Ap1 = hypre_StructMatrixBoxData(A, i, ssi[1]);
                     xoff1 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[1]]);

                  case 1:
                     Ap0 = hypre_StructMatrixBoxData(A, i, ssi[0]);
                     xoff0 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[ssi[0]]);

                  case 0:
                     break;
            }

            switch (depth)
            {
               case 27:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21] +
                               Ap22[Ai] * xp[xi + xoff22] +
                               Ap23[Ai] * xp[xi + xoff23] +
                               Ap24[Ai] * xp[xi + xoff24] +
                               Ap25[Ai] * xp[xi + xoff25] +
                               Ap26[Ai] * xp[xi + xoff26];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 26:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21] +
                               Ap22[Ai] * xp[xi + xoff22] +
                               Ap23[Ai] * xp[xi + xoff23] +
                               Ap24[Ai] * xp[xi + xoff24] +
                               Ap25[Ai] * xp[xi + xoff25];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 25:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21] +
                               Ap22[Ai] * xp[xi + xoff22] +
                               Ap23[Ai] * xp[xi + xoff23] +
                               Ap24[Ai] * xp[xi + xoff24];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 24:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21] +
                               Ap22[Ai] * xp[xi + xoff22] +
                               Ap23[Ai] * xp[xi + xoff23];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 23:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21] +
                               Ap22[Ai] * xp[xi + xoff22];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 22:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20] +
                               Ap21[Ai] * xp[xi + xoff21];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 21:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19] +
                               Ap20[Ai] * xp[xi + xoff20];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 20:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18] +
                               Ap19[Ai] * xp[xi + xoff19];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 19:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17] +
                               Ap18[Ai] * xp[xi + xoff18];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 18:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16] +
                               Ap17[Ai] * xp[xi + xoff17];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 17:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15] +
                               Ap16[Ai] * xp[xi + xoff16];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 16:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14] +
                               Ap15[Ai] * xp[xi + xoff15];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 15:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13] +
                               Ap14[Ai] * xp[xi + xoff14];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 14:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12] +
                               Ap13[Ai] * xp[xi + xoff13];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 13:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11] +
                               Ap12[Ai] * xp[xi + xoff12];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 12:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0]  +
                               Ap1[Ai]  * xp[xi + xoff1]  +
                               Ap2[Ai]  * xp[xi + xoff2]  +
                               Ap3[Ai]  * xp[xi + xoff3]  +
                               Ap4[Ai]  * xp[xi + xoff4]  +
                               Ap5[Ai]  * xp[xi + xoff5]  +
                               Ap6[Ai]  * xp[xi + xoff6]  +
                               Ap7[Ai]  * xp[xi + xoff7]  +
                               Ap8[Ai]  * xp[xi + xoff8]  +
                               Ap9[Ai]  * xp[xi + xoff9]  +
                               Ap10[Ai] * xp[xi + xoff10] +
                               Ap11[Ai] * xp[xi + xoff11];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 11:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5] +
                               Ap6[Ai]  * xp[xi + xoff6] +
                               Ap7[Ai]  * xp[xi + xoff7] +
                               Ap8[Ai]  * xp[xi + xoff8] +
                               Ap9[Ai]  * xp[xi + xoff9] +
                               Ap10[Ai] * xp[xi + xoff10];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 10:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5] +
                               Ap6[Ai]  * xp[xi + xoff6] +
                               Ap7[Ai]  * xp[xi + xoff7] +
                               Ap8[Ai]  * xp[xi + xoff8] +
                               Ap9[Ai]  * xp[xi + xoff9];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 9:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5] +
                               Ap6[Ai]  * xp[xi + xoff6] +
                               Ap7[Ai]  * xp[xi + xoff7] +
                               Ap8[Ai]  * xp[xi + xoff8];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 8:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5] +
                               Ap6[Ai]  * xp[xi + xoff6] +
                               Ap7[Ai]  * xp[xi + xoff7];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 7:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5] +
                               Ap6[Ai]  * xp[xi + xoff6];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 6:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4] +
                               Ap5[Ai]  * xp[xi + xoff5];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 5:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3] +
                               Ap4[Ai]  * xp[xi + xoff4];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 4:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2] +
                               Ap3[Ai]  * xp[xi + xoff3];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 3:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1] +
                               Ap2[Ai]  * xp[xi + xoff2];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 2:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0] +
                               Ap1[Ai]  * xp[xi + xoff1];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 1:
                  hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, ustride, Ai,
                                      x_data_box, start, ustride, xi,
                                      y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,xi,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop3For(Ai, xi, yi)
                  {
                     yp[yi] += Ap0[Ai]  * xp[xi + xoff0];
                  }
                  hypre_BoxLoop3End(Ai, xi, yi);
                  break;

               case 0:
                  break;
               }
            } /* loop on stencil entries */

            if (alpha != 1.0)
            {
               hypre_BoxLoop1Begin(ndim, loop_size,
                                   y_data_box, start, ustride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
               hypre_BoxLoop1For(yi)
               {
                  yp[yi] *= alpha;
               }
               hypre_BoxLoop1End(yi);
            }
         }
      }
   }

   if (x_tmp)
   {
      hypre_StructVectorDestroy(x_tmp);
      x = y;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
	hypre_StructMatvecData *matvec_data = (hypre_StructMatvecData *)matvec_vdata;

   if (matvec_data)
   {
      hypre_StructMatrixDestroy(matvec_data -> A);
      hypre_StructVectorDestroy(matvec_data -> x);
      hypre_ComputePkgDestroy(matvec_data -> compute_pkg);
      hypre_BoxArrayDestroy(matvec_data -> data_space);
      hypre_TFree(matvec_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvec( HYPRE_Complex       alpha,
                    hypre_StructMatrix *A,
                    hypre_StructVector *x,
                    HYPRE_Complex       beta,
                    hypre_StructVector *y )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecT( HYPRE_Complex       alpha,
                     hypre_StructMatrix *A,
                     hypre_StructVector *x,
                     HYPRE_Complex       beta,
                     hypre_StructVector *y )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetTranspose(matvec_data, 1);
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

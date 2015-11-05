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

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

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
   hypre_StructMatvecData  *matvec_data = matvec_vdata;
                          
   hypre_StructGrid        *grid;
   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;

   hypre_IndexRef           dom_stride;

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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The grids for A, x, and y must be compatible with respect to matrix-vector
 * multiply, but the initial grid boxes and strides may differ.  The routines
 * Reindex() and Resize() are called to convert the grid for the vector x to
 * match the domain grid of the matrix A.  As a result, both A and x have the
 * same list of underlying boxes and the domain stride for A is the same as the
 * stride for x.  The grid for y is assumed to match the range grid for A, but
 * with potentially different boxes and strides.  The box ids are used to find
 * the boxnums for y.  Here are some examples (after the Reindex() and Resize()
 * have been called for x):
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
hypre_StructMatvecCompute( void               *matvec_vdata,
                           HYPRE_Complex       alpha,
                           hypre_StructMatrix *A,
                           hypre_StructVector *x,
                           HYPRE_Complex       beta,
                           hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data = matvec_vdata;
                          
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
                          
   hypre_StructGrid        *base_grid;
   hypre_StructGrid        *ygrid;
   HYPRE_Int 		   *base_ids;
   HYPRE_Int 		   *y_ids;

   HYPRE_Int 		    ran_nboxes;
   HYPRE_Int 		   *ran_boxnums;
   hypre_IndexRef           ran_stride;
   hypre_IndexRef           dom_stride;
   HYPRE_Int                dom_is_coarse;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

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

   ygrid = hypre_StructVectorGrid(y);
   y_ids = hypre_StructGridIDs(ygrid);

   base_grid = hypre_StructMatrixGrid(A);
   base_ids = hypre_StructGridIDs(base_grid);

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
   hypre_StructVectorReindex(x, base_grid, dom_stride);
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

         Ab = ran_boxnums[i];
         xb = Ab;  /* Reindex ensures that A and x have the same grid boxes */
         if (y_ids[yb] > base_ids[Ab])
         {
            continue;
         }
         while (y_ids[yb] < base_ids[Ab])
         {
            yb++;
         }
         /* There should be a matching id for y */
         if (y_ids[yb] != base_ids[Ab])
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Matvec box ids don't match");
            return hypre_error_flag;
         }

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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
   hypre_StructMatvecData *matvec_data = matvec_vdata;

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


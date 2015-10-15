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
hypre_StructMatvecCreate( )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData, 1);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecSetup
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
   hypre_IndexRef           dom_stride;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;
   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);
   dom_stride = hypre_StructMatrixDomStride(A);

   /* This computes a data_space with respect to the matrix grid and the
    * stencil pattern for the matvec */
   hypre_StructVectorReindex(x, grid, dom_stride);
   hypre_StructNumGhostFromStencil(stencil, &num_ghost);
   hypre_StructVectorComputeDataSpace(x, num_ghost, &data_space);
   hypre_TFree(num_ghost);

   /* This computes the communication pattern for the new x data_space */
   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   if (hypre_StructMatrixDomainIsCoarse(A))
   {
       hypre_StructVectorMapCommInfo(x, hypre_ComputeInfoCommInfo(compute_info));
      /* Compute boxes will be appropriately projected in MatvecCompute */
   }
   hypre_ComputePkgCreate(compute_info, data_space, 1, grid, &compute_pkg);

   /* This restores the original grid */
   hypre_StructVectorRestore(x);

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
                           hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data = matvec_vdata;
                          
   hypre_ComputePkg        *compute_pkg;
                          
   hypre_CommHandle        *comm_handle;
                          
   hypre_BoxArray          *data_space;
   hypre_BoxArrayArray     *compute_box_aa;
   hypre_Box               *y_data_box;
                          
   HYPRE_Int                yi;
                          
   HYPRE_Complex           *xp;
   HYPRE_Complex           *yp;
                          
   hypre_BoxArray          *boxes;
   hypre_Box               *box;
   hypre_Index              loop_size, origin;
   hypre_IndexRef           start, stride;
                          
   HYPRE_Int                constant_coefficient;

   HYPRE_Complex            temp;
   HYPRE_Int                compute_i, i, j, si;

   HYPRE_Complex           *Ap;
   HYPRE_Int                xoff;
   HYPRE_Int                Ai;
   HYPRE_Int                xi;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;
                          
   hypre_Box               *A_data_box;
   hypre_Box               *x_data_box;
   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;
                          
   HYPRE_Int                ndim;
   HYPRE_Int 		    fi, ci;

   hypre_StructGrid        *base_grid;
   hypre_StructGrid        *ygrid;
   HYPRE_Int 		   *base_ids;
   HYPRE_Int 		   *y_ids;

   HYPRE_Int 		    ran_nboxes;
   HYPRE_Int 		   *ran_boxnums;
   hypre_IndexRef           ran_stride;

   HYPRE_Int 		    dom_nboxes;
   HYPRE_Int 		   *dom_boxnums;
   hypre_IndexRef           dom_stride;

   hypre_Index              unit_stride;
   hypre_IndexRef           y_stride;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) hypre_StructVectorClearBoundGhostValues(x, 0);

   compute_pkg = (matvec_data -> compute_pkg);

   stride = hypre_ComputePkgStride(compute_pkg);

   dom_stride = hypre_StructMatrixDomStride(A);
   dom_nboxes = hypre_StructMatrixDomNBoxes(A);
   dom_boxnums = hypre_StructMatrixDomBoxnums(A);

   ran_stride = hypre_StructMatrixRanStride(A);
   ran_nboxes = hypre_StructMatrixRanNBoxes(A);
   ran_boxnums = hypre_StructMatrixRanBoxnums(A);

   hypre_SetIndex(origin, 0);
   hypre_SetIndex(unit_stride, 1);

   y_stride = hypre_StructVectorStride(y);
   ygrid = hypre_StructVectorGrid(y);
   y_ids = hypre_StructGridIDs(ygrid);

   base_grid = hypre_StructMatrixGrid(A);
   base_ids = hypre_StructGridIDs(base_grid);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
      
      hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                             y_data_box, start, y_stride, yi);
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

   /* TODO: Create a vector on the finest grid (range or domain) with additional
    * ghost layers for communication, then copy the original x into that.
    * Alternatively (and maybe even better), write a "grow" and "shrink" routine
    * that will add or remove ghost layers from a given vector, possibly doing
    * it "in place" using ReAlloc() and careful reorganization of the data.  For
    * now, just assume that x has already been modified appropriately. */
   /* xorig = x; */
   /* create new x */
   /* copy xorig into x */

   /* This resizes the data for x using the data_space computed during setup */
   data_space = hypre_ComputePkgDataSpace(compute_pkg);
   hypre_StructVectorReindex(x, base_grid, dom_stride);
   hypre_StructVectorResize(x, data_space);


   ndim          = hypre_StructVectorNDim(x);
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
               boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
               hypre_ForBoxI(i, boxes)
               {
                  box   = hypre_BoxArrayBox(boxes, i);
                  start = hypre_BoxIMin(box);

                  y_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  yp = hypre_StructVectorBoxData(y, i);

                  if (temp == 0.0)
                  {
                     hypre_BoxGetSize(box, loop_size);

                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, y_stride, yi);
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
                     hypre_BoxGetSize(box, loop_size);

                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, y_stride, yi);
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
       * y += A*x
       *--------------------------------------------------------------------*/

      if (hypre_StructMatrixRangeIsCoarse(A))
      {
         ci = 0;
         for (i=0; i < ran_nboxes; i++) 
         /*hypre_ForBoxI(ci, cboxes)*/
         {
            hypre_IndexRef x_start;
            fi = ran_boxnums[i];
            /* This assumes that the grid boxes of y are a subset of the grid boxes 
		of A */
            while (y_ids[ci] > base_ids[fi]) 
            {
               i++;
               fi = ran_boxnums[i];
            }
            /*while (base_ids[fi] < y_ids[ci]) ci++; */
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);
            A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
            x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), fi);
            y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), ci);
            xp = hypre_StructVectorBoxData(x, fi);
            yp = hypre_StructVectorBoxData(y, ci);
            ran_stride = hypre_StructMatrixRanStride(A);
            dom_stride = hypre_StructMatrixDomStride(A);
            ci++;
            hypre_ForBoxI(j, compute_box_a)
            {
               hypre_Box *tmp_box;
               hypre_Index y_start;
               compute_box = hypre_BoxArrayBox(compute_box_a, j);
	       start  = hypre_BoxIMin(compute_box);
	       tmp_box = hypre_BoxCreate(ndim);
               hypre_CopyBox(compute_box, tmp_box);
               hypre_ProjectBox(tmp_box, origin, ran_stride);
               hypre_BoxGetStrideSize(tmp_box, ran_stride ,loop_size);
	       x_start  = hypre_BoxIMin(tmp_box);
               hypre_CopyIndex(x_start,y_start);
               hypre_SnapIndexNeg(y_start, origin, ran_stride, ndim);
               hypre_MapToCoarseIndex(y_start, origin, ran_stride, ndim);

               /* TODO (later, for optimization): Unroll these loops */
              for (si = 0; si < stencil_size; si++)
               {

                  Ap = hypre_StructMatrixBoxData(A, fi, si);
                  xoff = hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

                  if (hypre_StructMatrixConstEntry(A, si))
                  { 
                     /* Constant coefficient case */
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, x_start, ran_stride, xi,
                                      y_data_box, y_start, dom_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop2For(xi, yi)
                     {
                        yp[yi] += Ap[0] * xp[xi + xoff];
                     }
                     hypre_BoxLoop2End(xi, yi);
                  }
                  else
                  { 
                  /* Variable coefficient case */
                     hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, y_start, stride, Ai,
                                      x_data_box, x_start, ran_stride, xi,
                                      y_data_box, y_start, dom_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi,Ai) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop3For(Ai, xi, yi)
                     {
                        yp[yi] += Ap[Ai] * xp[xi + xoff];
                     }
                     hypre_BoxLoop3End(Ai, xi, yi);
                  }
               }

               if (alpha != 1.0)
               {
                  hypre_BoxLoop1Begin(ndim, loop_size,
                                   y_data_box, y_start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop1For(yi)
                  {
                     yp[yi] *= alpha;
                  }
                  hypre_BoxLoop1End(yi);
               }
	       hypre_BoxDestroy(tmp_box);
            }
         }
      }
      else 
      {
         ci = 0;
         for (i=0; i < ran_nboxes; i++) 
         {
            hypre_Index A_start;
            hypre_Index x_start;
            fi = ran_boxnums[i];
            /* This assumes that the grid boxes of y are a superset of the grid boxes 
		of A */
            while (y_ids[ci] < base_ids[fi]) 
            {
               ci++;
            }

            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

            A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
            y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), ci);
            x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), fi);

            xp = hypre_StructVectorBoxData(x, fi);
            yp = hypre_StructVectorBoxData(y, ci);

            hypre_ForBoxI(j, compute_box_a)
            {
               hypre_Box *tmp_box;
               compute_box = hypre_BoxArrayBox(compute_box_a, j);
               if (hypre_StructMatrixDomainIsCoarse(A))
	          tmp_box = hypre_BoxCreate(ndim);
               else
               {
                  hypre_BoxGetSize(compute_box, loop_size);
                  start  = hypre_BoxIMin(compute_box);
                  hypre_CopyIndex(start, x_start); 
                  hypre_CopyIndex(start, A_start); 
               }
               dom_stride = hypre_StructMatrixDomStride(A);
               ran_stride = hypre_StructMatrixRanStride(A);

               /* TODO (later, for optimization): Unroll these loops */
               for (si = 0; si < stencil_size; si++)
               {
                  /* If the the domain grid is coarse, loop over a subset of the
                   * range compute box based on the current stencil entry */
                  if (hypre_StructMatrixDomainIsCoarse(A))
                  {
                     hypre_CopyBox(compute_box, tmp_box);
                     hypre_BoxShiftNeg(tmp_box, stencil_shape[si]);
                     hypre_ProjectBox(tmp_box, origin, dom_stride);
                     start = hypre_BoxIMin(tmp_box);
                     hypre_CopyIndex(start, x_start); 
                     hypre_StructVectorMapDataIndex(x, x_start);
                     hypre_BoxShiftPos(tmp_box, stencil_shape[si]);
                     hypre_BoxGetStrideSize(tmp_box, dom_stride, loop_size);
                     hypre_CopyIndex(start, A_start);
                     hypre_StructMatrixMapDataIndex(A, A_start);
                  }
                  Ap = hypre_StructMatrixBoxData(A, fi, si);
                  xoff = hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);
                  if (hypre_StructMatrixConstEntry(A, si))
                  {
                     /* Constant coefficient case */
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                  x_data_box, x_start, ran_stride, xi,
                                  y_data_box, start, dom_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop2For(xi, yi)
                     {
                        yp[yi] += Ap[0] * xp[xi + xoff];
                     }
                     hypre_BoxLoop2End(xi, yi);
                  }
                  else
                  {
                     /* Variable coefficient case */
                     hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, A_start, stride, Ai,
                                   x_data_box, x_start, ran_stride, xi,
                                   y_data_box, start, dom_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi,xi,Ai) HYPRE_SMP_SCHEDULE
#endif
                     hypre_BoxLoop3For(Ai, xi, yi)
                     {
                        yp[yi] += Ap[Ai] * xp[xi + xoff];
                     }
                     hypre_BoxLoop3End(Ai, xi, yi);
                  }
               }

               if (alpha != 1.0)
               {
                  hypre_BoxLoop1Begin(ndim, loop_size,
                                y_data_box, start, stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,yi) HYPRE_SMP_SCHEDULE
#endif
                  hypre_BoxLoop1For(yi)
                  {
                     yp[yi] *= alpha;
                  }
                  hypre_BoxLoop1End(yi);
               }
               if (hypre_StructMatrixDomainIsCoarse(A))
	          hypre_BoxDestroy(tmp_box);
            }
         }
      }
   }

   if (x_tmp)
   {
      hypre_StructVectorDestroy(x_tmp);
      x = y;
   }

   /* This restores the original grid and data layout */
   hypre_StructVectorRestore(x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
   hypre_StructMatvecData *matvec_data = matvec_vdata;

   if (matvec_data)
   {
      hypre_StructMatrixDestroy(matvec_data -> A);
      hypre_StructVectorDestroy(matvec_data -> x);
      hypre_ComputePkgDestroy(matvec_data -> compute_pkg );
      hypre_TFree(matvec_data);
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
                    hypre_StructVector *y )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

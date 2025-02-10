/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Cyclic reduction algorithm (coded as if it were a 1D MG method)
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define hypre_CycRedSetCIndex(base_index, base_stride, level, cdir, cindex) \
   {                                                                    \
      if (level > 0)                                                    \
         hypre_SetIndex3(cindex, 0, 0, 0);                              \
      else                                                              \
         hypre_CopyIndex(base_index,  cindex);                          \
      hypre_IndexD(cindex, cdir) += 0;                                  \
   }

#define hypre_CycRedSetFIndex(base_index, base_stride, level, cdir, findex) \
   {                                                                    \
      if (level > 0)                                                    \
         hypre_SetIndex3(findex, 0, 0, 0);                              \
      else                                                              \
         hypre_CopyIndex(base_index,  findex);                          \
      hypre_IndexD(findex, cdir) += 1;                                  \
   }

#define hypre_CycRedSetStride(base_index, base_stride, level, cdir, stride) \
   {                                                                    \
      if (level > 0)                                                    \
         hypre_SetIndex3(stride, 1, 1, 1);                              \
      else                                                              \
         hypre_CopyIndex(base_stride, stride);                          \
      hypre_IndexD(stride, cdir) *= 2;                                  \
   }

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   HYPRE_Int             num_levels;

   HYPRE_Int             ndim;
   HYPRE_Int             cdir;         /* coarsening direction */
   hypre_Index           base_index;
   hypre_Index           base_stride;

   hypre_StructGrid    **grid_l;

   hypre_BoxArray       *base_points;
   hypre_BoxArray      **fine_points_l;

   HYPRE_MemoryLocation  memory_location; /* memory location of data */
   HYPRE_Real           *data;
   HYPRE_Real           *data_const;
   hypre_StructMatrix  **A_l;
   hypre_StructVector  **x_l;

   hypre_ComputePkg    **down_compute_pkg_l;
   hypre_ComputePkg    **up_compute_pkg_l;

   HYPRE_Int             time_index;
   HYPRE_BigInt          solve_flops;
   HYPRE_Int             max_levels;
} hypre_CyclicReductionData;

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CyclicReductionCreate( MPI_Comm  comm )
{
   hypre_CyclicReductionData *cyc_red_data;

   cyc_red_data = hypre_CTAlloc(hypre_CyclicReductionData,  1, HYPRE_MEMORY_HOST);

   (cyc_red_data -> comm) = comm;
   (cyc_red_data -> ndim) = 3;
   (cyc_red_data -> cdir) = 0;
   (cyc_red_data -> time_index)  = hypre_InitializeTiming("CyclicReduction");
   (cyc_red_data -> max_levels)  = -1;

   /* set defaults */
   hypre_SetIndex3((cyc_red_data -> base_index), 0, 0, 0);
   hypre_SetIndex3((cyc_red_data -> base_stride), 1, 1, 1);

   (cyc_red_data -> memory_location) = hypre_HandleMemoryLocation(hypre_handle());

   return (void *) cyc_red_data;
}

/*--------------------------------------------------------------------------
 * hypre_CycRedCreateCoarseOp
 *
 * NOTE: This routine assumes that domain boundary ghost zones (i.e., ghost
 * zones that do not intersect the grid) have the identity equation in them.
 * This is currently insured by the MatrixAssemble routine.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_CycRedCreateCoarseOp( hypre_StructMatrix *A,
                            hypre_StructGrid   *coarse_grid,
                            HYPRE_Int           cdir        )
{
   HYPRE_Int              ndim = hypre_StructMatrixNDim(A);
   hypre_StructMatrix    *Ac;
   hypre_Index           *Ac_stencil_shape;
   hypre_StructStencil   *Ac_stencil;
   HYPRE_Int              Ac_stencil_size;
   HYPRE_Int              Ac_num_ghost[] = {0, 0, 0, 0, 0, 0};

   HYPRE_Int              i;
   HYPRE_Int              stencil_rank;

   /*-----------------------------------------------
    * Define Ac_stencil
    *-----------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------
    * non-symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *-----------------------------------------------*/

   if (!hypre_StructMatrixSymmetric(A))
   {
      Ac_stencil_size = 3;
      Ac_stencil_shape = hypre_CTAlloc(hypre_Index,  Ac_stencil_size, HYPRE_MEMORY_HOST);
      for (i = -1; i < 2; i++)
      {
         /* Storage for 3 elements (c,w,e) */
         hypre_SetIndex3(Ac_stencil_shape[stencil_rank], 0, 0, 0);
         hypre_IndexD(Ac_stencil_shape[stencil_rank], cdir) = i;
         stencil_rank++;
      }
   }

   /*-----------------------------------------------
    * symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *
    * Only store the lower triangular part + diagonal = 2 entries,
    * lower triangular means the lower triangular part on the matrix
    * in the standard lexicalgraphic ordering.
    *-----------------------------------------------*/

   else
   {
      Ac_stencil_size = 2;
      Ac_stencil_shape = hypre_CTAlloc(hypre_Index,  Ac_stencil_size, HYPRE_MEMORY_HOST);
      for (i = -1; i < 1; i++)
      {

         /* Storage for 2 elements in (c,w) */
         hypre_SetIndex3(Ac_stencil_shape[stencil_rank], 0, 0, 0);
         hypre_IndexD(Ac_stencil_shape[stencil_rank], cdir) = i;
         stencil_rank++;
      }
   }

   Ac_stencil = hypre_StructStencilCreate(ndim, Ac_stencil_size, Ac_stencil_shape);

   Ac = hypre_StructMatrixCreate(hypre_StructMatrixComm(A),
                                 coarse_grid, Ac_stencil);

   hypre_StructStencilDestroy(Ac_stencil);

   /*-----------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------*/

   hypre_StructMatrixSymmetric(Ac) = hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------
    * Set number of ghost points
    *-----------------------------------------------*/

   Ac_num_ghost[2 * cdir] = 1;
   if (!hypre_StructMatrixSymmetric(A))
   {
      Ac_num_ghost[2 * cdir + 1] = 1;
   }
   hypre_StructMatrixSetNumGhost(Ac, Ac_num_ghost);

   hypre_StructMatrixInitializeShell(Ac);

   return Ac;
}

/*--------------------------------------------------------------------------
 * hypre_CycRedSetupCoarseOp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CycRedSetupCoarseOp( hypre_StructMatrix *A,
                           hypre_StructMatrix *Ac,
                           hypre_Index         cindex,
                           hypre_Index         cstride,
                           HYPRE_Int           cdir )
{
   hypre_Index             index;

   hypre_StructGrid       *fgrid;
   HYPRE_Int              *fgrid_ids;
   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   HYPRE_Int              *cgrid_ids;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             fstart;
   hypre_IndexRef          stridef;
   hypre_Index             loop_size;

   HYPRE_Int               fi, ci;

   hypre_Box              *A_dbox;
   hypre_Box              *Ac_dbox;

   HYPRE_Real             *a_cc, *a_cw, *a_ce;
   HYPRE_Real             *ac_cc, *ac_cw, *ac_ce = NULL;

   HYPRE_Int               offsetA;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(Ac);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

      cstart = hypre_BoxIMin(cgrid_box);
      hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
      Ac_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Ac), ci);

      /*-----------------------------------------------
       * Extract pointers for 3-point fine grid operator:
       *
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient
       * a_ce is pointer for east coefficient
       *-----------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_IndexD(index, cdir) = -1;
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_IndexD(index, cdir) = 1;
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      /*-----------------------------------------------
       * Extract pointers for coarse grid operator - always 3-point:
       *
       * If A is symmetric so is Ac.  We build only the
       * lower triangular part (plus diagonal).
       *
       * ac_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      ac_cc = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

      hypre_IndexD(index, cdir) = -1;
      ac_cw = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_IndexD(index, cdir) = 1;
         ac_ce = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
      }

      /*-----------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer
       * to data associated with the point which we are
       * building the stencil for.  The below offsets
       * are used in refering to data associated with
       * other points.
       *-----------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      hypre_IndexD(index, cdir) = 1;
      offsetA = hypre_BoxOffsetDistance(A_dbox, index);

      /*-----------------------------------------------
       * non-symmetric case
       *-----------------------------------------------*/

      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cw,a_cw,a_cc,ac_cc,a_ce,ac_ce)
         hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, fstart, stridef, iA,
                             Ac_dbox, cstart, stridec, iAc);
         {
            HYPRE_Int iAm1 = iA - offsetA;
            HYPRE_Int iAp1 = iA + offsetA;

            ac_cw[iAc] = -a_cw[iA] * a_cw[iAm1] / a_cc[iAm1];

            ac_cc[iAc] = a_cc[iA] - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1] -
                         a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];

            ac_ce[iAc] = -a_ce[iA] * a_ce[iAp1] / a_cc[iAp1];

         }
         hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR
      }

      /*-----------------------------------------------
       * symmetric case
       *-----------------------------------------------*/

      else
      {
         hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cw,a_cw,a_cc,ac_cc,a_ce)
         hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, fstart, stridef, iA,
                             Ac_dbox, cstart, stridec, iAc);
         {
            HYPRE_Int iAm1 = iA - offsetA;
            HYPRE_Int iAp1 = iA + offsetA;

            ac_cw[iAc] = -a_cw[iA] * a_cw[iAm1] / a_cc[iAm1];

            ac_cc[iAc] = a_cc[iA] - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1] -
                         a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];
         }
         hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR
      }

   } /* end ForBoxI */

   hypre_StructMatrixAssemble(Ac);

   /*-----------------------------------------------------------------------
    * Collapse stencil in periodic direction on coarsest grid.
    *-----------------------------------------------------------------------*/

   if (hypre_IndexD(hypre_StructGridPeriodic(cgrid), cdir) == 1)
   {
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = hypre_BoxIMin(cgrid_box);

         Ac_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Ac), ci);

         /*-----------------------------------------------
          * Extract pointers for coarse grid operator - always 3-point:
          *
          * If A is symmetric so is Ac.  We build only the
          * lower triangular part (plus diagonal).
          *
          * ac_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------*/

         hypre_SetIndex3(index, 0, 0, 0);
         ac_cc = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         hypre_IndexD(index, cdir) = -1;
         ac_cw = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         if (!hypre_StructMatrixSymmetric(A))
         {
            hypre_IndexD(index, cdir) = 1;
            ac_ce = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
         }

         /*-----------------------------------------------
          * non-symmetric case
          *-----------------------------------------------*/

         if (!hypre_StructMatrixSymmetric(A))
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cc,ac_cw,ac_ce)
            hypre_BoxLoop1Begin(hypre_StructMatrixNDim(A), loop_size,
                                Ac_dbox, cstart, stridec, iAc);
            {
               ac_cc[iAc] += (ac_cw[iAc] + ac_ce[iAc]);
               ac_cw[iAc]  =  0.0;
               ac_ce[iAc]  =  0.0;
            }
            hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

         /*-----------------------------------------------
          * symmetric case
          *-----------------------------------------------*/

         else
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cc,ac_cw)
            hypre_BoxLoop1Begin(hypre_StructMatrixNDim(A), loop_size,
                                Ac_dbox, cstart, stridec, iAc);
            {
               ac_cc[iAc] += (2.0 * ac_cw[iAc]);
               ac_cw[iAc]  =  0.0;
            }
            hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

      } /* end ForBoxI */

   }

   hypre_StructMatrixAssemble(Ac);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReductionSetup( void               *cyc_red_vdata,
                            hypre_StructMatrix *A,
                            hypre_StructVector *b,
                            hypre_StructVector *x             )
{
   HYPRE_UNUSED_VAR(b);

   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *) cyc_red_vdata;

   MPI_Comm                comm        = (cyc_red_data -> comm);
   HYPRE_Int               cdir        = (cyc_red_data -> cdir);
   hypre_IndexRef          base_index  = (cyc_red_data -> base_index);
   hypre_IndexRef          base_stride = (cyc_red_data -> base_stride);

   HYPRE_Int               num_levels;
   HYPRE_Int               max_levels = -1;
   hypre_StructGrid      **grid_l;
   hypre_BoxArray         *base_points;
   hypre_BoxArray        **fine_points_l;
   HYPRE_Real             *data;
   HYPRE_Real             *data_const;
   HYPRE_Int               data_size = 0;
   HYPRE_Int               data_size_const = 0;
   hypre_StructMatrix    **A_l;
   hypre_StructVector    **x_l;
   hypre_ComputePkg      **down_compute_pkg_l;
   hypre_ComputePkg      **up_compute_pkg_l;
   hypre_ComputeInfo      *compute_info;

   hypre_Index             cindex;
   hypre_Index             findex;
   hypre_Index             stride;

   hypre_StructGrid       *grid;
   hypre_Box              *cbox;
   HYPRE_Int               l;
   HYPRE_Int               flop_divisor;
   HYPRE_Int               x_num_ghost[] = {0, 0, 0, 0, 0, 0};

   HYPRE_MemoryLocation    memory_location = hypre_StructMatrixMemoryLocation(A);

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid = hypre_StructMatrixGrid(A);

   /* Compute a preliminary num_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
   num_levels = hypre_Log2(hypre_BoxSizeD(cbox, cdir)) + 2;
   if (cyc_red_data -> max_levels > 0)
   {
      max_levels = (cyc_red_data -> max_levels);
   }


   grid_l    = hypre_TAlloc(hypre_StructGrid *,  num_levels, HYPRE_MEMORY_HOST);
   hypre_StructGridRef(grid, &grid_l[0]);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   data_location = hypre_StructGridDataLocation(grid);
#endif
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      if ( hypre_BoxIMinD(cbox, cdir) == hypre_BoxIMaxD(cbox, cdir) ||
           (l == (max_levels - 1)))
      {
         /* stop coarsening */
         break;
      }

      /* coarsen cbox */
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                  hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                  hypre_BoxIMax(cbox));

      /* coarsen the grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_StructGridDataLocation(grid_l[l + 1]) = data_location;
#endif
   }
   num_levels = l + 1;

   /* free up some things */
   hypre_BoxDestroy(cbox);

   (cyc_red_data -> ndim)            = hypre_StructGridNDim(grid);
   (cyc_red_data -> num_levels)      = num_levels;
   (cyc_red_data -> grid_l)          = grid_l;

   /*-----------------------------------------------------
    * Set up base points
    *-----------------------------------------------------*/

   base_points = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[0]));
   hypre_ProjectBoxArray(base_points, base_index, base_stride);

   (cyc_red_data -> base_points) = base_points;

   /*-----------------------------------------------------
    * Set up fine points
    *-----------------------------------------------------*/

   fine_points_l   = hypre_TAlloc(hypre_BoxArray *,   num_levels, HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      fine_points_l[l] = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[l]));
      hypre_ProjectBoxArray(fine_points_l[l], findex, stride);
   }

   fine_points_l[l] = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[l]));
   if (num_levels == 1)
   {
      hypre_ProjectBoxArray(fine_points_l[l], base_index, base_stride);
   }

   (cyc_red_data -> fine_points_l)   = fine_points_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *,  num_levels, HYPRE_MEMORY_HOST);
   x_l  = hypre_TAlloc(hypre_StructVector *,  num_levels, HYPRE_MEMORY_HOST);

   A_l[0] = hypre_StructMatrixRef(A);
   x_l[0] = hypre_StructVectorRef(x);

   x_num_ghost[2 * cdir]     = 1;
   x_num_ghost[2 * cdir + 1] = 1;

   for (l = 0; l < (num_levels - 1); l++)
   {
      A_l[l + 1] = hypre_CycRedCreateCoarseOp(A_l[l], grid_l[l + 1], cdir);
      //hypre_StructMatrixInitializeShell(A_l[l+1]);
      data_size += hypre_StructMatrixDataSize(A_l[l + 1]);
      data_size_const += hypre_StructMatrixDataConstSize(A_l[l + 1]);

      x_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetNumGhost(x_l[l + 1], x_num_ghost);
      hypre_StructVectorInitializeShell(x_l[l + 1]);
      hypre_StructVectorSetDataSize(x_l[l + 1], &data_size, &data_size_const);
   }

   data = hypre_CTAlloc(HYPRE_Real, data_size, memory_location);
   data_const = hypre_CTAlloc(HYPRE_Real, data_size_const, HYPRE_MEMORY_HOST);

   (cyc_red_data -> memory_location) = memory_location;
   (cyc_red_data -> data) = data;
   (cyc_red_data -> data_const) = data_const;

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_StructMatrixInitializeData(A_l[l + 1], data, data_const);
      data += hypre_StructMatrixDataSize(A_l[l + 1]);
      data_const += hypre_StructMatrixDataConstSize(A_l[l + 1]);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (data_location != HYPRE_MEMORY_HOST)
      {
         hypre_StructVectorInitializeData(x_l[l + 1], data);
         hypre_StructVectorAssemble(x_l[l + 1]);
         data += hypre_StructVectorDataSize(x_l[l + 1]);
      }
      else
      {
         hypre_StructVectorInitializeData(x_l[l + 1], data_const);
         hypre_StructVectorAssemble(x_l[l + 1]);
         data_const += hypre_StructVectorDataSize(x_l[l + 1]);
      }
#else
      hypre_StructVectorInitializeData(x_l[l + 1], data);
      hypre_StructVectorAssemble(x_l[l + 1]);
      data += hypre_StructVectorDataSize(x_l[l + 1]);
#endif
   }

   (cyc_red_data -> A_l)  = A_l;
   (cyc_red_data -> x_l)  = x_l;

   /*-----------------------------------------------------
    * Set up coarse grid operators
    *-----------------------------------------------------*/

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      hypre_CycRedSetupCoarseOp(A_l[l], A_l[l + 1], cindex, stride, cdir);
   }

   /*----------------------------------------------------------
    * Set up compute packages
    *----------------------------------------------------------*/

   down_compute_pkg_l = hypre_TAlloc(hypre_ComputePkg *,  (num_levels - 1), HYPRE_MEMORY_HOST);
   up_compute_pkg_l   = hypre_TAlloc(hypre_ComputePkg *,  (num_levels - 1), HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* down-cycle */
      hypre_CreateComputeInfo(grid_l[l], hypre_StructMatrixStencil(A_l[l]),
                              &compute_info);
      hypre_ComputeInfoProjectSend(compute_info, findex, stride);
      hypre_ComputeInfoProjectRecv(compute_info, findex, stride);
      hypre_ComputeInfoProjectComp(compute_info, cindex, stride);
      hypre_ComputePkgCreate(compute_info,
                             hypre_StructVectorDataSpace(x_l[l]), 1,
                             grid_l[l], &down_compute_pkg_l[l]);

      /* up-cycle */
      hypre_CreateComputeInfo(grid_l[l], hypre_StructMatrixStencil(A_l[l]),
                              &compute_info);
      hypre_ComputeInfoProjectSend(compute_info, cindex, stride);
      hypre_ComputeInfoProjectRecv(compute_info, cindex, stride);
      hypre_ComputeInfoProjectComp(compute_info, findex, stride);
      hypre_ComputePkgCreate(compute_info,
                             hypre_StructVectorDataSpace(x_l[l]), 1,
                             grid_l[l], &up_compute_pkg_l[l]);
   }

   (cyc_red_data -> down_compute_pkg_l) = down_compute_pkg_l;
   (cyc_red_data -> up_compute_pkg_l)   = up_compute_pkg_l;

   /*-----------------------------------------------------
    * Compute solve flops
    *-----------------------------------------------------*/

   flop_divisor = (hypre_IndexX(base_stride) *
                   hypre_IndexY(base_stride) *
                   hypre_IndexZ(base_stride)  );
   (cyc_red_data -> solve_flops) =
      hypre_StructVectorGlobalSize(x_l[0]) / 2 / (HYPRE_BigInt)flop_divisor;
   (cyc_red_data -> solve_flops) +=
      5 * hypre_StructVectorGlobalSize(x_l[0]) / 2 / (HYPRE_BigInt)flop_divisor;
   for (l = 1; l < (num_levels - 1); l++)
   {
      (cyc_red_data -> solve_flops) +=
         10 * hypre_StructVectorGlobalSize(x_l[l]) / 2;
   }

   if (num_levels > 1)
   {
      (cyc_red_data -> solve_flops) +=
         hypre_StructVectorGlobalSize(x_l[l]) / 2;
   }


   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

#if DEBUG
   {
      char  filename[255];

      /* debugging stuff */
      for (l = 0; l < num_levels; l++)
      {
         hypre_sprintf(filename, "yout_A.%02d", l);
         hypre_StructMatrixPrint(filename, A_l[l], 0);
      }
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReduction
 *
 * The solution vectors on each level are also used to store the
 * right-hand-side data.  We can do this because of the red-black
 * nature of the algorithm and the fact that the method is exact,
 * allowing one to assume initial guesses of zero on all grid levels.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReduction( void               *cyc_red_vdata,
                       hypre_StructMatrix *A,
                       hypre_StructVector *b,
                       hypre_StructVector *x             )
{
   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *)cyc_red_vdata;

   HYPRE_Int             num_levels      = (cyc_red_data -> num_levels);
   HYPRE_Int             cdir            = (cyc_red_data -> cdir);
   hypre_IndexRef        base_index      = (cyc_red_data -> base_index);
   hypre_IndexRef        base_stride     = (cyc_red_data -> base_stride);
   hypre_BoxArray       *base_points     = (cyc_red_data -> base_points);
   hypre_BoxArray      **fine_points_l   = (cyc_red_data -> fine_points_l);
   hypre_StructMatrix  **A_l             = (cyc_red_data -> A_l);
   hypre_StructVector  **x_l             = (cyc_red_data -> x_l);
   hypre_ComputePkg    **down_compute_pkg_l = (cyc_red_data -> down_compute_pkg_l);
   hypre_ComputePkg    **up_compute_pkg_l   = (cyc_red_data -> up_compute_pkg_l);

   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;

   hypre_CommHandle     *comm_handle;

   hypre_BoxArrayArray  *compute_box_aa;
   hypre_BoxArray       *compute_box_a;
   hypre_Box            *compute_box;

   hypre_Box            *A_dbox;
   hypre_Box            *x_dbox;
   hypre_Box            *b_dbox;
   hypre_Box            *xc_dbox;

   HYPRE_Real           *Ap, *Awp, *Aep;
   HYPRE_Real           *xp, *xwp, *xep;
   HYPRE_Real           *bp;
   HYPRE_Real           *xcp;

   hypre_Index           cindex;
   hypre_Index           stride;

   hypre_Index           index;
   hypre_Index           loop_size;
   hypre_Index           start;
   hypre_Index           startc;
   hypre_Index           stridec;

   HYPRE_Int             compute_i, fi, ci, j, l;

   hypre_BeginTiming(cyc_red_data -> time_index);


   /*--------------------------------------------------
    * Initialize some things
    *--------------------------------------------------*/

   hypre_SetIndex3(stridec, 1, 1, 1);

   hypre_StructMatrixDestroy(A_l[0]);
   hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   x_l[0] = hypre_StructVectorRef(x);

   /*--------------------------------------------------
    * Copy b into x
    *--------------------------------------------------*/

   compute_box_a = base_points;
   hypre_ForBoxI(fi, compute_box_a)
   {
      compute_box = hypre_BoxArrayBox(compute_box_a, fi);

      x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), fi);
      b_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), fi);

      xp = hypre_StructVectorBoxData(x, fi);
      bp = hypre_StructVectorBoxData(b, fi);

      hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
      hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,bp)
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_dbox, start, base_stride, xi,
                          b_dbox, start, base_stride, bi);
      {
         xp[xi] = bp[bi];
      }
      hypre_BoxLoop2End(xi, bi);
#undef DEVICE_VAR
   }

   /*--------------------------------------------------
    * Down cycle:
    *
    * 1) Do an F-relaxation sweep with zero initial guess
    * 2) Compute and inject residual at C-points
    *    - computations are at C-points
    *    - communications are at F-points
    *
    * Notes:
    * - Before these two steps are executed, the
    * fine-grid solution vector contains the right-hand-side.
    * - After these two steps are executed, the fine-grid
    * solution vector contains the right-hand side at
    * C-points and the current solution approximation at
    * F-points.  The coarse-grid solution vector contains
    * the restricted (injected) fine-grid residual.
    *--------------------------------------------------*/

   for (l = 0; l < num_levels - 1 ; l++)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      compute_box_a = fine_points_l[l];
      hypre_ForBoxI(fi, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, fi);

         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
         x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);

         hypre_SetIndex3(index, 0, 0, 0);
         Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
         xp = hypre_StructVectorBoxData(x_l[l], fi);

         hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
         hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Ap)
         hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                             A_dbox, start, stride, Ai,
                             x_dbox, start, stride, xi);
         {
            xp[xi] /= Ap[Ai];
         }
         hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
      }

      /* Step 2 */
      fgrid = hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = hypre_StructGridIDs(fgrid);
      cgrid = hypre_StructVectorGrid(x_l[l + 1]);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      cgrid_ids = hypre_StructGridIDs(cgrid);

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               hypre_InitializeIndtComputations(down_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa = hypre_ComputePkgIndtBoxes(down_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = hypre_ComputePkgDeptBoxes(down_compute_pkg_l[l]);
            }
            break;
         }

         fi = 0;
         hypre_ForBoxI(ci, cgrid_boxes)
         {
            while (fgrid_ids[fi] != cgrid_ids[ci])
            {
               fi++;
            }

            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

            A_dbox  = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
            x_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);
            xc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l + 1]), ci);

            xp  = hypre_StructVectorBoxData(x_l[l], fi);
            xcp = hypre_StructVectorBoxData(x_l[l + 1], ci);

            hypre_SetIndex3(index, 0, 0, 0);
            hypre_IndexD(index, cdir) = -1;
            Awp = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xwp = hypre_StructVectorBoxData(x_l[l], fi);
            //RL:PTR_OFFSET
            HYPRE_Int xwp_offset = hypre_BoxOffsetDistance(x_dbox, index);

            hypre_SetIndex3(index, 0, 0, 0);
            hypre_IndexD(index, cdir) = 1;
            Aep = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xep = hypre_StructVectorBoxData(x_l[l], fi);
            HYPRE_Int xep_offset = hypre_BoxOffsetDistance(x_dbox, index);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
               hypre_StructMapFineToCoarse(start, cindex, stride, startc);

               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xcp,xp,Awp,xwp,Aep,xep)
               hypre_BoxLoop3Begin(hypre_StructVectorNDim(x), loop_size,
                                   A_dbox, start, stride, Ai,
                                   x_dbox, start, stride, xi,
                                   xc_dbox, startc, stridec, xci);
               {
                  xcp[xci] = xp[xi] - Awp[Ai] * xwp[xi + xwp_offset] -
                             Aep[Ai] * xep[xi + xep_offset];
               }
               hypre_BoxLoop3End(Ai, xi, xci);
#undef DEVICE_VAR
            }
         }
      }
   }
   /*--------------------------------------------------
    * Coarsest grid:
    *
    * Do an F-relaxation sweep with zero initial guess
    *
    * This is the same as step 1 in above, but is
    * broken out as a sepecial case to add a check
    * for zero diagonal that can occur for singlar
    * problems like the full Neumann problem.
    *--------------------------------------------------*/
   /* set cindex and stride */
   hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
   hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

   compute_box_a = fine_points_l[l];
   hypre_ForBoxI(fi, compute_box_a)
   {
      compute_box = hypre_BoxArrayBox(compute_box_a, fi);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
      x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);

      hypre_SetIndex3(index, 0, 0, 0);
      Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
      xp = hypre_StructVectorBoxData(x_l[l], fi);

      hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
      hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Ap)
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          A_dbox, start, stride, Ai,
                          x_dbox, start, stride, xi);
      {
         if (Ap[Ai] != 0.0)
         {
            xp[xi] /= Ap[Ai];
         }
      }
      hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
   }

   /*--------------------------------------------------
    * Up cycle:
    *
    * 1) Inject coarse error into fine-grid solution
    *    vector (this is the solution at the C-points)
    * 2) Do an F-relaxation sweep on Ax = 0 and update
    *    solution at F-points
    *    - computations are at F-points
    *    - communications are at C-points
    *--------------------------------------------------*/

   for (l = (num_levels - 2); l >= 0; l--)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      fgrid = hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = hypre_StructGridIDs(fgrid);
      cgrid = hypre_StructVectorGrid(x_l[l + 1]);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      cgrid_ids = hypre_StructGridIDs(cgrid);

      fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         compute_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         hypre_CopyIndex(hypre_BoxIMin(compute_box), startc);
         hypre_StructMapCoarseToFine(startc, cindex, stride, start);

         x_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);
         xc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l + 1]), ci);

         xp  = hypre_StructVectorBoxData(x_l[l], fi);
         xcp = hypre_StructVectorBoxData(x_l[l + 1], ci);

         hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,xcp)
         hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                             x_dbox, start, stride, xi,
                             xc_dbox, startc, stridec, xci);
         {
            xp[xi] = xcp[xci];
         }
         hypre_BoxLoop2End(xi, xci);
#undef DEVICE_VAR
      }

      /* Step 2 */
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               hypre_InitializeIndtComputations(up_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa = hypre_ComputePkgIndtBoxes(up_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = hypre_ComputePkgDeptBoxes(up_compute_pkg_l[l]);
            }
            break;
         }

         hypre_ForBoxArrayI(fi, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

            A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
            x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);

            hypre_SetIndex3(index, 0, 0, 0);
            Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xp = hypre_StructVectorBoxData(x_l[l], fi);

            hypre_SetIndex3(index, 0, 0, 0);
            hypre_IndexD(index, cdir) = -1;
            Awp = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            //RL PTROFFSET
            xwp = hypre_StructVectorBoxData(x_l[l], fi);
            HYPRE_Int xwp_offset = hypre_BoxOffsetDistance(x_dbox, index);

            hypre_SetIndex3(index, 0, 0, 0);
            hypre_IndexD(index, cdir) = 1;
            Aep = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xep = hypre_StructVectorBoxData(x_l[l], fi);
            HYPRE_Int xep_offset = hypre_BoxOffsetDistance(x_dbox, index);

            hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, j);

               hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Awp,Aep,Ap)
               hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                                   A_dbox, start, stride, Ai,
                                   x_dbox, start, stride, xi);
               {
                  xp[xi] -= (Awp[Ai] * xp[xi + xwp_offset] + Aep[Ai] * xp[xi + xep_offset]) / Ap[Ai];
               }
               hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
            }
         }
      }
   }

   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

   hypre_IncFLOPCount(cyc_red_data -> solve_flops);
   hypre_EndTiming(cyc_red_data -> time_index);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionSetBase
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReductionSetBase( void        *cyc_red_vdata,
                              hypre_Index  base_index,
                              hypre_Index  base_stride )
{
   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *)cyc_red_vdata;
   HYPRE_Int                d;

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((cyc_red_data -> base_index),  d) =
         hypre_IndexD(base_index,  d);
      hypre_IndexD((cyc_red_data -> base_stride), d) =
         hypre_IndexD(base_stride, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionSetCDir
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReductionSetCDir( void        *cyc_red_vdata,
                              HYPRE_Int    cdir )
{
   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *)cyc_red_vdata;

   (cyc_red_data -> cdir) = cdir;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReductionDestroy( void *cyc_red_vdata )
{
   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *)cyc_red_vdata;

   HYPRE_Int l;

   if (cyc_red_data)
   {
      HYPRE_MemoryLocation memory_location = cyc_red_data -> memory_location;

      hypre_BoxArrayDestroy(cyc_red_data -> base_points);
      hypre_StructGridDestroy(cyc_red_data -> grid_l[0]);
      hypre_StructMatrixDestroy(cyc_red_data -> A_l[0]);
      hypre_StructVectorDestroy(cyc_red_data -> x_l[0]);
      for (l = 0; l < ((cyc_red_data -> num_levels) - 1); l++)
      {
         hypre_StructGridDestroy(cyc_red_data -> grid_l[l + 1]);
         hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
         hypre_StructMatrixDestroy(cyc_red_data -> A_l[l + 1]);
         hypre_StructVectorDestroy(cyc_red_data -> x_l[l + 1]);
         hypre_ComputePkgDestroy(cyc_red_data -> down_compute_pkg_l[l]);
         hypre_ComputePkgDestroy(cyc_red_data -> up_compute_pkg_l[l]);
      }
      hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
      hypre_TFree(cyc_red_data -> data, memory_location);
      hypre_TFree(cyc_red_data -> grid_l, HYPRE_MEMORY_HOST);
      hypre_TFree(cyc_red_data -> fine_points_l, HYPRE_MEMORY_HOST);
      hypre_TFree(cyc_red_data -> A_l, HYPRE_MEMORY_HOST);
      hypre_TFree(cyc_red_data -> x_l, HYPRE_MEMORY_HOST);
      hypre_TFree(cyc_red_data -> down_compute_pkg_l, HYPRE_MEMORY_HOST);
      hypre_TFree(cyc_red_data -> up_compute_pkg_l, HYPRE_MEMORY_HOST);

      hypre_FinalizeTiming(cyc_red_data -> time_index);
      hypre_TFree(cyc_red_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CyclicReductionSetMaxLevel( void   *cyc_red_vdata,
                                  HYPRE_Int   max_level  )
{
   hypre_CyclicReductionData *cyc_red_data = (hypre_CyclicReductionData *)cyc_red_vdata;
   (cyc_red_data -> max_levels) = max_level;

   return hypre_error_flag;
}

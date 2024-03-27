/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *R;
   HYPRE_Int           R_stored_as_transpose;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         stride;

   HYPRE_Int           time_index;

} hypre_SemiRestrictData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SemiRestrictCreate( void )
{
   hypre_SemiRestrictData *restrict_data;

   restrict_data = hypre_CTAlloc(hypre_SemiRestrictData,  1, HYPRE_MEMORY_HOST);

   (restrict_data -> time_index)  = hypre_InitializeTiming("SemiRestrict");

   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiRestrictSetup( void               *restrict_vdata,
                         hypre_StructMatrix *R,
                         HYPRE_Int           R_stored_as_transpose,
                         hypre_StructVector *r,
                         hypre_StructVector *rc,
                         hypre_Index         cindex,
                         hypre_Index         findex,
                         hypre_Index         stride                )
{
   HYPRE_UNUSED_VAR(rc);

   hypre_SemiRestrictData *restrict_data = (hypre_SemiRestrictData *)restrict_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;

   hypre_ComputeInfo      *compute_info;
   hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(r);
   stencil = hypre_StructMatrixStencil(R);

   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_ComputeInfoProjectSend(compute_info, findex, stride);
   hypre_ComputeInfoProjectRecv(compute_info, findex, stride);
   hypre_ComputeInfoProjectComp(compute_info, cindex, stride);
   hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(r), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the restrict data structure
    *----------------------------------------------------------*/

   (restrict_data -> R) = hypre_StructMatrixRef(R);
   (restrict_data -> R_stored_as_transpose) = R_stored_as_transpose;
   (restrict_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex, (restrict_data -> cindex));
   hypre_CopyIndex(stride, (restrict_data -> stride));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiRestrict( void               *restrict_vdata,
                    hypre_StructMatrix *R,
                    hypre_StructVector *r,
                    hypre_StructVector *rc             )
{
   hypre_SemiRestrictData *restrict_data = (hypre_SemiRestrictData *)restrict_vdata;

   HYPRE_Int               R_stored_as_transpose;
   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          stride;

   hypre_StructGrid       *fgrid;
   HYPRE_Int              *fgrid_ids;
   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   HYPRE_Int              *cgrid_ids;

   hypre_CommHandle       *comm_handle;

   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;

   hypre_Box              *R_dbox;
   hypre_Box              *r_dbox;
   hypre_Box              *rc_dbox;

   HYPRE_Int               Ri;
   HYPRE_Int               constant_coefficient;

   HYPRE_Real             *Rp0, *Rp1;
   HYPRE_Real             *rp;
   HYPRE_Real             *rcp;

   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_Index             startc;
   hypre_Index             stridec;

   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   HYPRE_Int               compute_i, fi, ci, j;
   hypre_StructVector     *rc_tmp;
   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   hypre_BeginTiming(restrict_data -> time_index);

   R_stored_as_transpose = (restrict_data -> R_stored_as_transpose);
   compute_pkg   = (restrict_data -> compute_pkg);
   cindex        = (restrict_data -> cindex);
   stride        = (restrict_data -> stride);

   stencil       = hypre_StructMatrixStencil(R);
   stencil_shape = hypre_StructStencilShape(stencil);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(R);
   hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   /* ... if A has constant_coefficient==2, R has constant_coefficient==0 */

   if (constant_coefficient) { hypre_StructVectorClearBoundGhostValues(r, 0); }

   hypre_SetIndex3(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Restrict the residual.
    *--------------------------------------------------------------------*/

   fgrid = hypre_StructVectorGrid(r);
   fgrid_ids = hypre_StructGridIDs(fgrid);
   cgrid = hypre_StructVectorGrid(rc);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_MemoryLocation data_location_f = hypre_StructGridDataLocation(fgrid);
   HYPRE_MemoryLocation data_location_c = hypre_StructGridDataLocation(cgrid);

   if (data_location_f != data_location_c)
   {
      rc_tmp = hypre_StructVectorCreate(hypre_MPI_COMM_WORLD, cgrid);
      hypre_StructVectorSetNumGhost(rc_tmp, hypre_StructVectorNumGhost(rc));
      hypre_StructGridDataLocation(cgrid) = data_location_f;
      hypre_StructVectorInitialize(rc_tmp);
      hypre_StructVectorAssemble(rc_tmp);
   }
   else
   {
      rc_tmp = rc;
   }
#else
   rc_tmp = rc;
#endif

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            rp = hypre_StructVectorData(r);
            hypre_InitializeIndtComputations(compute_pkg, rp, &comm_handle);
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

      fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

         R_dbox  = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R),  fi);
         r_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r),  fi);
         rc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(rc), ci);

         // RL: PTROFFSET
         HYPRE_Int Rp0_offset = 0, rp0_offset, rp1_offset;

         if (R_stored_as_transpose)
         {
            if ( constant_coefficient )
            {
               Rp0 = hypre_StructMatrixBoxData(R, fi, 1);
               Rp1 = hypre_StructMatrixBoxData(R, fi, 0);
               Rp0_offset = -hypre_CCBoxOffsetDistance(R_dbox, stencil_shape[1]);
            }
            else
            {
               Rp0 = hypre_StructMatrixBoxData(R, fi, 1);
               Rp1 = hypre_StructMatrixBoxData(R, fi, 0);
               Rp0_offset = -hypre_BoxOffsetDistance(R_dbox, stencil_shape[1]);
            }
         }
         else
         {
            Rp0 = hypre_StructMatrixBoxData(R, fi, 0);
            Rp1 = hypre_StructMatrixBoxData(R, fi, 1);
         }
         rp  = hypre_StructVectorBoxData(r, fi);
         rp0_offset = hypre_BoxOffsetDistance(r_dbox, stencil_shape[0]);
         rp1_offset = hypre_BoxOffsetDistance(r_dbox, stencil_shape[1]);
         rcp = hypre_StructVectorBoxData(rc_tmp, ci);

         hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = hypre_BoxArrayBox(compute_box_a, j);

            start  = hypre_BoxIMin(compute_box);
            hypre_StructMapFineToCoarse(start, cindex, stride, startc);

            hypre_BoxGetStrideSize(compute_box, stride, loop_size);

            if ( constant_coefficient )
            {
               HYPRE_Complex Rp0val, Rp1val;
               Ri = hypre_CCBoxIndexRank( R_dbox, startc );

               Rp0val = Rp0[Ri + Rp0_offset];
               Rp1val = Rp1[Ri];
#define DEVICE_VAR is_device_ptr(rcp,rp)
               hypre_BoxLoop2Begin(hypre_StructMatrixNDim(R), loop_size,
                                   r_dbox,  start,  stride,  ri,
                                   rc_dbox, startc, stridec, rci);
               {
                  rcp[rci] = rp[ri] + (Rp0val * rp[ri + rp0_offset] +
                                       Rp1val * rp[ri + rp1_offset]);
               }
               hypre_BoxLoop2End(ri, rci);
#undef DEVICE_VAR
            }
            else
            {
#define DEVICE_VAR is_device_ptr(rcp,rp,Rp0,Rp1)
               hypre_BoxLoop3Begin(hypre_StructMatrixNDim(R), loop_size,
                                   R_dbox,  startc, stridec, Ri,
                                   r_dbox,  start,  stride,  ri,
                                   rc_dbox, startc, stridec, rci);
               {
                  rcp[rci] = rp[ri] + (Rp0[Ri + Rp0_offset] * rp[ri + rp0_offset] +
                                       Rp1[Ri]            * rp[ri + rp1_offset]);
               }
               hypre_BoxLoop3End(Ri, ri, rci);
#undef DEVICE_VAR
            }
         }
      }
   }
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (data_location_f != data_location_c)
   {
      hypre_TMemcpy(hypre_StructVectorData(rc), hypre_StructVectorData(rc_tmp), HYPRE_Complex,
                    hypre_StructVectorDataSize(rc_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_StructVectorDestroy(rc_tmp);
      hypre_StructGridDataLocation(cgrid) = data_location_c;
   }
#endif
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(4 * hypre_StructVectorGlobalSize(rc));
   hypre_EndTiming(restrict_data -> time_index);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiRestrictDestroy( void *restrict_vdata )
{
   hypre_SemiRestrictData *restrict_data = (hypre_SemiRestrictData *)restrict_vdata;

   if (restrict_data)
   {
      hypre_StructMatrixDestroy(restrict_data -> R);
      hypre_ComputePkgDestroy(restrict_data -> compute_pkg);
      hypre_FinalizeTiming(restrict_data -> time_index);
      hypre_TFree(restrict_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

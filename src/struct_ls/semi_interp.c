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
   hypre_StructMatrix *P;
   HYPRE_Int           P_stored_as_transpose;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         findex;
   hypre_Index         stride;

   HYPRE_Int           time_index;

} hypre_SemiInterpData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SemiInterpCreate( void )
{
   hypre_SemiInterpData *interp_data;

   interp_data = hypre_CTAlloc(hypre_SemiInterpData,  1, HYPRE_MEMORY_HOST);
   (interp_data -> time_index)  = hypre_InitializeTiming("SemiInterp");

   return (void *) interp_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiInterpSetup( void               *interp_vdata,
                       hypre_StructMatrix *P,
                       HYPRE_Int           P_stored_as_transpose,
                       hypre_StructVector *xc,
                       hypre_StructVector *e,
                       hypre_Index         cindex,
                       hypre_Index         findex,
                       hypre_Index         stride       )
{
   HYPRE_UNUSED_VAR(xc);

   hypre_SemiInterpData   *interp_data = (hypre_SemiInterpData   *)interp_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;

   hypre_ComputeInfo      *compute_info;
   hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(e);
   stencil = hypre_StructMatrixStencil(P);

   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_ComputeInfoProjectSend(compute_info, cindex, stride);
   hypre_ComputeInfoProjectRecv(compute_info, cindex, stride);
   hypre_ComputeInfoProjectComp(compute_info, findex, stride);
   hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(e), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the interp data structure
    *----------------------------------------------------------*/

   (interp_data -> P) = hypre_StructMatrixRef(P);
   (interp_data -> P_stored_as_transpose) = P_stored_as_transpose;
   (interp_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex, (interp_data -> cindex));
   hypre_CopyIndex(findex, (interp_data -> findex));
   hypre_CopyIndex(stride, (interp_data -> stride));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiInterp( void               *interp_vdata,
                  hypre_StructMatrix *P,
                  hypre_StructVector *xc,
                  hypre_StructVector *e            )
{
   hypre_SemiInterpData   *interp_data = (hypre_SemiInterpData   *)interp_vdata;

   HYPRE_Int               P_stored_as_transpose;
   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          findex;
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

   hypre_Box              *P_dbox;
   hypre_Box              *xc_dbox;
   hypre_Box              *e_dbox;

   HYPRE_Int               Pi;
   HYPRE_Int               constant_coefficient;

   HYPRE_Real             *Pp0, *Pp1;
   HYPRE_Real             *xcp;
   HYPRE_Real             *ep;

   hypre_Index             loop_size;
   hypre_Index             start;
   hypre_Index             startc;
   hypre_Index             stridec;

   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   HYPRE_Int               compute_i, fi, ci, j;
   hypre_StructVector     *xc_tmp;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   hypre_BeginTiming(interp_data -> time_index);

   P_stored_as_transpose = (interp_data -> P_stored_as_transpose);
   compute_pkg   = (interp_data -> compute_pkg);
   cindex        = (interp_data -> cindex);
   findex        = (interp_data -> findex);
   stride        = (interp_data -> stride);

   stencil       = hypre_StructMatrixStencil(P);
   stencil_shape = hypre_StructStencilShape(stencil);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(P);
   hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   /* ... constant_coefficient==2 for P shouldn't happen, see
      hypre_PFMGCreateInterpOp in pfmg_setup_interp.c */

   if (constant_coefficient) { hypre_StructVectorClearBoundGhostValues(e, 0); }

   hypre_SetIndex3(stridec, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection)
    *-----------------------------------------------------------------------*/

   fgrid = hypre_StructVectorGrid(e);
   fgrid_ids = hypre_StructGridIDs(fgrid);
   cgrid = hypre_StructVectorGrid(xc);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_MemoryLocation data_location_f = hypre_StructGridDataLocation(fgrid);
   HYPRE_MemoryLocation data_location_c = hypre_StructGridDataLocation(cgrid);

   if (data_location_f != data_location_c)
   {
      xc_tmp = hypre_StructVectorCreate(hypre_MPI_COMM_WORLD, cgrid);
      hypre_StructVectorSetNumGhost(xc_tmp, hypre_StructVectorNumGhost(xc));
      hypre_StructGridDataLocation(cgrid) = data_location_f;
      hypre_StructVectorInitialize(xc_tmp);
      hypre_StructVectorAssemble(xc_tmp);
      hypre_TMemcpy(hypre_StructVectorData(xc_tmp), hypre_StructVectorData(xc), HYPRE_Complex,
                    hypre_StructVectorDataSize(xc), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   else
   {
      xc_tmp = xc;
   }
#else
   xc_tmp = xc;
#endif
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

      e_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), fi);
      xc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), ci);

      ep  = hypre_StructVectorBoxData(e, fi);
      xcp = hypre_StructVectorBoxData(xc_tmp, ci);

      hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(ep,xcp)
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(P), loop_size,
                          e_dbox, start, stride, ei,
                          xc_dbox, startc, stridec, xci);
      {
         ep[ei] = xcp[xci];
      }
      hypre_BoxLoop2End(ei, xci);
#undef DEVICE_VAR
   }

   /*-----------------------------------------------------------------------
    * Compute e at fine points
    *-----------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            ep = hypre_StructVectorData(e);
            hypre_InitializeIndtComputations(compute_pkg, ep, &comm_handle);
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

      hypre_ForBoxArrayI(fi, compute_box_aa)
      {
         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

         P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
         e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), fi);

         //RL:PTROFFSET
         HYPRE_Int Pp1_offset = 0, ep0_offset, ep1_offset;
         if (P_stored_as_transpose)
         {
            if ( constant_coefficient )
            {
               Pp0 = hypre_StructMatrixBoxData(P, fi, 1);
               Pp1 = hypre_StructMatrixBoxData(P, fi, 0);
               Pp1_offset = -hypre_CCBoxOffsetDistance(P_dbox, stencil_shape[0]);
            }
            else
            {
               Pp0 = hypre_StructMatrixBoxData(P, fi, 1);
               Pp1 = hypre_StructMatrixBoxData(P, fi, 0);
               Pp1_offset = -hypre_BoxOffsetDistance(P_dbox, stencil_shape[0]);
            }
         }
         else
         {
            Pp0 = hypre_StructMatrixBoxData(P, fi, 0);
            Pp1 = hypre_StructMatrixBoxData(P, fi, 1);
         }
         ep  = hypre_StructVectorBoxData(e, fi);
         ep0_offset = hypre_BoxOffsetDistance(e_dbox, stencil_shape[0]);
         ep1_offset = hypre_BoxOffsetDistance(e_dbox, stencil_shape[1]);

         hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = hypre_BoxArrayBox(compute_box_a, j);

            hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
            hypre_StructMapFineToCoarse(start, findex, stride, startc);

            hypre_BoxGetStrideSize(compute_box, stride, loop_size);

            if ( constant_coefficient )
            {
               HYPRE_Complex Pp0val, Pp1val;
               Pi = hypre_CCBoxIndexRank( P_dbox, startc );
               Pp0val = Pp0[Pi];
               Pp1val = Pp1[Pi + Pp1_offset];

#define DEVICE_VAR is_device_ptr(ep)
               hypre_BoxLoop1Begin(hypre_StructMatrixNDim(P), loop_size,
                                   e_dbox, start, stride, ei);
               {
                  ep[ei] =  (Pp0val * ep[ei + ep0_offset] +
                             Pp1val * ep[ei + ep1_offset]);
               }
               hypre_BoxLoop1End(ei);
#undef DEVICE_VAR
            }
            else
            {
#define DEVICE_VAR is_device_ptr(ep,Pp0,Pp1)
               hypre_BoxLoop2Begin(hypre_StructMatrixNDim(P), loop_size,
                                   P_dbox, startc, stridec, Pi,
                                   e_dbox, start, stride, ei);
               {
                  ep[ei] =  (Pp0[Pi]            * ep[ei + ep0_offset] +
                             Pp1[Pi + Pp1_offset] * ep[ei + ep1_offset]);
               }
               hypre_BoxLoop2End(Pi, ei);
#undef DEVICE_VAR
            }
         }
      }
   }
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (data_location_f != data_location_c)
   {
      hypre_StructVectorDestroy(xc_tmp);
      hypre_StructGridDataLocation(cgrid) = data_location_c;
   }
#endif
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(3 * hypre_StructVectorGlobalSize(xc));
   hypre_EndTiming(interp_data -> time_index);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiInterpDestroy( void *interp_vdata )
{
   hypre_SemiInterpData *interp_data = (hypre_SemiInterpData   *)interp_vdata;

   if (interp_data)
   {
      hypre_StructMatrixDestroy(interp_data -> P);
      hypre_ComputePkgDestroy(interp_data -> compute_pkg);
      hypre_FinalizeTiming(interp_data -> time_index);
      hypre_TFree(interp_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

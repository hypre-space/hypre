/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.19 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SemiRestrictData data structure
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
 * hypre_SemiRestrictCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SemiRestrictCreate( )
{
   hypre_SemiRestrictData *restrict_data;

   restrict_data = hypre_CTAlloc(hypre_SemiRestrictData, 1);

   (restrict_data -> time_index)  = hypre_InitializeTiming("SemiRestrict");
   
   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 * hypre_SemiRestrictSetup
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
   hypre_SemiRestrictData *restrict_data = restrict_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;

   hypre_ComputeInfo      *compute_info;
   hypre_ComputePkg       *compute_pkg;

   HYPRE_Int               ierr = 0;

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
   hypre_CopyIndex(cindex ,(restrict_data -> cindex));
   hypre_CopyIndex(stride ,(restrict_data -> stride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SemiRestrict:
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiRestrict( void               *restrict_vdata,
                    hypre_StructMatrix *R,
                    hypre_StructVector *r,
                    hypre_StructVector *rc             )
{
   HYPRE_Int ierr = 0;

   hypre_SemiRestrictData *restrict_data = restrict_vdata;

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
   HYPRE_Int               ri;
   HYPRE_Int               rci;
   HYPRE_Int               constant_coefficient;

   double                 *Rp0, *Rp1;
   double                 *rp, *rp0, *rp1;
   double                 *rcp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_Index             startc;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   HYPRE_Int               compute_i, fi, ci, j;
   HYPRE_Int               loopi, loopj, loopk;

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
   hypre_assert( constant_coefficient==0 || constant_coefficient==1 );
   /* ... if A has constant_coefficient==2, R has constant_coefficient==0 */

   if (constant_coefficient) hypre_StructVectorClearBoundGhostValues(r, 0);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Restrict the residual.
    *--------------------------------------------------------------------*/

   fgrid = hypre_StructVectorGrid(r);
   fgrid_ids = hypre_StructGridIDs(fgrid);
   cgrid = hypre_StructVectorGrid(rc);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
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

            if (R_stored_as_transpose)
            {
               if ( constant_coefficient )
               {
                  Rp0 = hypre_StructMatrixBoxData(R, fi, 1) -
                     hypre_CCBoxOffsetDistance(R_dbox, stencil_shape[1]);
                  Rp1 = hypre_StructMatrixBoxData(R, fi, 0);
               }
               else
               {
                  Rp0 = hypre_StructMatrixBoxData(R, fi, 1) -
                     hypre_BoxOffsetDistance(R_dbox, stencil_shape[1]);
                  Rp1 = hypre_StructMatrixBoxData(R, fi, 0);
               }
            }
            else
            {
               Rp0 = hypre_StructMatrixBoxData(R, fi, 0);
               Rp1 = hypre_StructMatrixBoxData(R, fi, 1);
            }
            rp  = hypre_StructVectorBoxData(r, fi);
            rp0 = rp + hypre_BoxOffsetDistance(r_dbox, stencil_shape[0]);
            rp1 = rp + hypre_BoxOffsetDistance(r_dbox, stencil_shape[1]);
            rcp = hypre_StructVectorBoxData(rc, ci);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);
                  hypre_StructMapFineToCoarse(start, cindex, stride, startc);

                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                  if ( constant_coefficient )
                  {
                     Ri = hypre_CCBoxIndexRank( R_dbox, startc );

                     hypre_BoxLoop2Begin(loop_size,
                                         r_dbox,  start,  stride,  ri,
                                         rc_dbox, startc, stridec, rci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,ri,rci
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop2For(loopi, loopj, loopk, ri, rci)
                        {
                           rcp[rci] = rp[ri] + (Rp0[Ri] * rp0[ri] +
                                                Rp1[Ri] * rp1[ri]);
                        }
                     hypre_BoxLoop2End(ri, rci);
                  }
                  else
                  {
                     hypre_BoxLoop3Begin(loop_size,
                                         R_dbox,  startc, stridec, Ri,
                                         r_dbox,  start,  stride,  ri,
                                         rc_dbox, startc, stridec, rci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ri,ri,rci
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ri, ri, rci)
                        {
                           rcp[rci] = rp[ri] + (Rp0[Ri] * rp0[ri] +
                                                Rp1[Ri] * rp1[ri]);
                        }
                     hypre_BoxLoop3End(Ri, ri, rci);
                  }
               }
         }
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(4*hypre_StructVectorGlobalSize(rc));
   hypre_EndTiming(restrict_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SemiRestrictDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiRestrictDestroy( void *restrict_vdata )
{
   HYPRE_Int ierr = 0;

   hypre_SemiRestrictData *restrict_data = restrict_vdata;

   if (restrict_data)
   {
      hypre_StructMatrixDestroy(restrict_data -> R);
      hypre_ComputePkgDestroy(restrict_data -> compute_pkg);
      hypre_FinalizeTiming(restrict_data -> time_index);
      hypre_TFree(restrict_data);
   }

   return ierr;
}


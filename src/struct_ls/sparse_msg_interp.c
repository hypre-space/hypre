/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SparseMSGInterpData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *P;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         findex;
   hypre_Index         stride;
   hypre_Index         strideP;

   HYPRE_Int           time_index;

} hypre_SparseMSGInterpData;

/*--------------------------------------------------------------------------
 * hypre_SparseMSGInterpCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SparseMSGInterpCreate( )
{
   hypre_SparseMSGInterpData *interp_data;

   interp_data = hypre_CTAlloc(hypre_SparseMSGInterpData, 1);
   (interp_data -> time_index)  = hypre_InitializeTiming("SparseMSGInterp");

   return (void *) interp_data;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGInterpSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGInterpSetup( void               *interp_vdata,
                            hypre_StructMatrix *P,
                            hypre_StructVector *xc,
                            hypre_StructVector *e,
                            hypre_Index         cindex,
                            hypre_Index         findex,
                            hypre_Index         stride,
                            hypre_Index         strideP       )
{
   hypre_SparseMSGInterpData   *interp_data = interp_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_ComputeInfo      *compute_info;
   hypre_ComputePkg       *compute_pkg;

   HYPRE_Int               ierr = 0;

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
   (interp_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex, (interp_data -> cindex));
   hypre_CopyIndex(findex, (interp_data -> findex));
   hypre_CopyIndex(stride, (interp_data -> stride));
   hypre_CopyIndex(strideP, (interp_data -> strideP));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGInterp:
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGInterp( void               *interp_vdata,
                       hypre_StructMatrix *P,
                       hypre_StructVector *xc,
                       hypre_StructVector *e            )
{
   HYPRE_Int ierr = 0;

   hypre_SparseMSGInterpData   *interp_data = interp_vdata;

   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          findex;
   hypre_IndexRef          stride;
   hypre_IndexRef          strideP;

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
   HYPRE_Int               xci;
   HYPRE_Int               ei;
                         
   double                 *Pp0, *Pp1;
   double                 *xcp;
   double                 *ep, *ep0, *ep1;
                       
   hypre_Index             loop_size;
   hypre_Index             start;
   hypre_Index             startc;
   hypre_Index             startP;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   HYPRE_Int               compute_i, fi, ci, j;
   HYPRE_Int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   hypre_BeginTiming(interp_data -> time_index);

   compute_pkg   = (interp_data -> compute_pkg);
   cindex        = (interp_data -> cindex);
   findex        = (interp_data -> findex);
   stride        = (interp_data -> stride);
   strideP       = (interp_data -> strideP);

   stencil       = hypre_StructMatrixStencil(P);
   stencil_shape = hypre_StructStencilShape(stencil);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection)
    *-----------------------------------------------------------------------*/

   fgrid = hypre_StructVectorGrid(e);
   fgrid_ids = hypre_StructGridIDs(fgrid);
   cgrid = hypre_StructVectorGrid(xc);
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

         e_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), fi);
         xc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), ci);

         ep  = hypre_StructVectorBoxData(e, fi);
         xcp = hypre_StructVectorBoxData(xc, ci);

         hypre_BoxGetSize(compute_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             e_dbox,  start,  stride,  ei,
                             xc_dbox, startc, stridec, xci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,ei,xci
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, ei, xci)
            {
               ep[ei] = xcp[xci];
            }
         hypre_BoxLoop2End(ei, xci);
      }

   /*-----------------------------------------------------------------------
    * Compute e at fine points
    *-----------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
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

            Pp0 = hypre_StructMatrixBoxData(P, fi, 0);
            Pp1 = hypre_StructMatrixBoxData(P, fi, 1);
            ep  = hypre_StructVectorBoxData(e, fi);
            ep0 = ep + hypre_BoxOffsetDistance(e_dbox, stencil_shape[0]);
            ep1 = ep + hypre_BoxOffsetDistance(e_dbox, stencil_shape[1]);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
                  hypre_StructMapFineToCoarse(start,  findex, stride,  startc);
                  hypre_StructMapCoarseToFine(startc, cindex, strideP, startP);

                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                  hypre_BoxLoop2Begin(loop_size,
                                      P_dbox, startP, strideP, Pi,
                                      e_dbox, start,  stride,  ei);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Pi,ei
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop2For(loopi, loopj, loopk, Pi, ei)
                     {
                        ep[ei] =  (Pp0[Pi] * ep0[ei] +
                                   Pp1[Pi] * ep1[ei]);
                     }
                  hypre_BoxLoop2End(Pi, ei);
               }
         }
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(3*hypre_StructVectorGlobalSize(xc));
   hypre_EndTiming(interp_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGInterpDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGInterpDestroy( void *interp_vdata )
{
   HYPRE_Int ierr = 0;

   hypre_SparseMSGInterpData *interp_data = interp_vdata;

   if (interp_data)
   {
      hypre_StructMatrixDestroy(interp_data -> P);
      hypre_ComputePkgDestroy(interp_data -> compute_pkg);
      hypre_FinalizeTiming(interp_data -> time_index);
      hypre_TFree(interp_data);
   }

   return ierr;
}


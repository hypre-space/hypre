/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif
#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_PFMGInterpData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *P;
   hypre_ComputePkg   *compute_pkg;
   hypre_BoxArray     *coarse_points;
   hypre_Index         cindex;
   hypre_Index         findex;
   hypre_Index         stride;

   int                 time_index;

} hypre_PFMGInterpData;

/*--------------------------------------------------------------------------
 * hypre_PFMGInterpInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_PFMGInterpInitialize( )
{
   hypre_PFMGInterpData *interp_data;

   interp_data = hypre_CTAlloc(hypre_PFMGInterpData, 1);
   (interp_data -> time_index)  = hypre_InitializeTiming("PFMGInterp");

   return (void *) interp_data;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGInterpSetup
 *--------------------------------------------------------------------------*/

int
hypre_PFMGInterpSetup( void               *interp_vdata,
                       hypre_StructMatrix *P,
                       hypre_StructVector *xc,
                       hypre_StructVector *e,
                       hypre_Index         cindex,
                       hypre_Index         findex,
                       hypre_Index         stride       )
{
   hypre_PFMGInterpData   *interp_data = interp_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_ComputePkg       *compute_pkg;
   hypre_BoxArray         *coarse_points;

   int                     ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(e);
   stencil = hypre_StructMatrixStencil(P);

   hypre_GetComputeInfo(grid, stencil,
                        &send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes);

   hypre_ProjectBoxArrayArray(send_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(recv_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(indt_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(dept_boxes, findex, stride);

   hypre_NewComputePkg(send_boxes, recv_boxes,
                       stride, stride,
                       send_processes, recv_processes,
                       indt_boxes, dept_boxes,
                       stride, grid,
                       hypre_StructVectorDataSpace(e), 1,
                       &compute_pkg);

   /*----------------------------------------------------------
    * Set up the coarse points BoxArray
    *----------------------------------------------------------*/

   coarse_points = hypre_DuplicateBoxArray(hypre_StructGridBoxes(grid));
   hypre_ProjectBoxArray(coarse_points, cindex, stride);

   /*----------------------------------------------------------
    * Set up the interp data structure
    *----------------------------------------------------------*/

   (interp_data -> P)             = P;
   (interp_data -> compute_pkg)   = compute_pkg;
   (interp_data -> coarse_points) = coarse_points;
   hypre_CopyIndex(cindex, (interp_data -> cindex));
   hypre_CopyIndex(findex, (interp_data -> findex));
   hypre_CopyIndex(stride, (interp_data -> stride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGInterp:
 *--------------------------------------------------------------------------*/

int
hypre_PFMGInterp( void               *interp_vdata,
                  hypre_StructMatrix *P,
                  hypre_StructVector *xc,
                  hypre_StructVector *e            )
{
   int ierr = 0;

   hypre_PFMGInterpData    *interp_data = interp_vdata;

   hypre_ComputePkg       *compute_pkg;
   hypre_BoxArray         *coarse_points;
   hypre_IndexRef          cindex;
   hypre_IndexRef          findex;
   hypre_IndexRef          stride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *P_data_box;
   hypre_Box              *xc_data_box;
   hypre_Box              *x_data_box;
   hypre_Box              *e_data_box;
                       
   int                     Pi;
   int                     xci;
   int                     xi;
   int                     ei;
                         
   double                 *Pp0, *Pp1;
   double                 *xcp;
   double                 *ep, *ep0, *ep1;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_Index             startc;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   int                     compute_i, i, j;
   int                     loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   hypre_BeginTiming(interp_data -> time_index);

   compute_pkg   = (interp_data -> compute_pkg);
   coarse_points = (interp_data -> coarse_points);
   cindex        = (interp_data -> cindex);
   findex        = (interp_data -> findex);
   stride        = (interp_data -> stride);

   stencil       = hypre_StructMatrixStencil(P);
   stencil_shape = hypre_StructStencilShape(stencil);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection)
    *-----------------------------------------------------------------------*/

   compute_box_a = coarse_points;
   hypre_ForBoxI(i, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, i);

         start = hypre_BoxIMin(compute_box);
         hypre_PFMGMapFineToCoarse(start, cindex, stride, startc);

         e_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
         xc_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), i);

         ep  = hypre_StructVectorBoxData(e, i);
         xcp = hypre_StructVectorBoxData(xc, i);

         hypre_GetStrideBoxSize(compute_box, stride, loop_size);
         hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                        e_data_box,  start,  stride,  ei,
                        xc_data_box, startc, stridec, xci,
                        {
                           ep[ei] = xcp[xci];
                        });
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

      hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            P_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);
            e_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);

            Pp0 = hypre_StructMatrixBoxData(P, i, 0);
            Pp1 = hypre_StructMatrixBoxData(P, i, 1);
            ep  = hypre_StructVectorBoxData(e, i);
            ep0 = ep + hypre_BoxOffsetDistance(e_data_box, stencil_shape[0]);
            ep1 = ep + hypre_BoxOffsetDistance(e_data_box, stencil_shape[1]);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);
                  hypre_PFMGMapFineToCoarse(start, findex, stride, startc);

                  hypre_GetStrideBoxSize(compute_box, stride, loop_size);
                  hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                                 P_data_box, startc, stridec, Pi,
                                 e_data_box, start,  stride,  ei,
                                 {
                                    ep[ei] = (Pp0[Pi] * ep0[ei] +
                                              Pp1[Pi] * ep1[ei]);
                                 });
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
 * hypre_PFMGInterpFinalize
 *--------------------------------------------------------------------------*/

int
hypre_PFMGInterpFinalize( void *interp_vdata )
{
   int ierr = 0;

   hypre_PFMGInterpData *interp_data = interp_vdata;

   if (interp_data)
   {
      hypre_FreeBoxArray(interp_data -> coarse_points);
      hypre_FreeComputePkg(interp_data -> compute_pkg);
      hypre_FinalizeTiming(interp_data -> time_index);
      hypre_TFree(interp_data);
   }

   return ierr;
}


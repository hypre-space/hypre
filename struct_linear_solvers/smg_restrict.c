/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
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
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGRestrictData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *R;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         stride;

   int                 time_index;

} hypre_SMGRestrictData;

/*--------------------------------------------------------------------------
 * hypre_SMGRestrictInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_SMGRestrictInitialize( )
{
   hypre_SMGRestrictData *restrict_data;

   restrict_data = hypre_CTAlloc(hypre_SMGRestrictData, 1);

   (restrict_data -> time_index)  = hypre_InitializeTiming("SMGRestrict");
   
   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRestrictSetup
 *--------------------------------------------------------------------------*/

int
hypre_SMGRestrictSetup( void               *restrict_vdata,
                        hypre_StructMatrix *R,
                        hypre_StructVector *r,
                        hypre_StructVector *rc,
                        hypre_Index         cindex,
                        hypre_Index         findex,
                        hypre_Index         stride            )
{
   hypre_SMGRestrictData  *restrict_data = restrict_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_ComputePkg       *compute_pkg;

   int                     ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(r);
   stencil = hypre_StructMatrixStencil(R);

   hypre_GetComputeInfo(grid, stencil,
                        &send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes);

   hypre_ProjectBoxArrayArray(send_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(recv_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(indt_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(dept_boxes, cindex, stride);

   hypre_NewComputePkg(send_boxes, recv_boxes,
                       stride, stride,
                       send_processes, recv_processes,
                       indt_boxes, dept_boxes,
                       stride, grid,
                       hypre_StructVectorDataSpace(r), 1,
                       &compute_pkg);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (restrict_data -> R)           = R;
   (restrict_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex ,(restrict_data -> cindex));
   hypre_CopyIndex(stride ,(restrict_data -> stride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRestrict:
 *
 * Notes:
 *    This routine assumes that the interpolation stencil is a
 *    2-point stencil.
 *--------------------------------------------------------------------------*/

int
hypre_SMGRestrict( void               *restrict_vdata,
                   hypre_StructMatrix *R,
                   hypre_StructVector *r,
                   hypre_StructVector *rc             )
{
   int ierr = 0;

   hypre_SMGRestrictData  *restrict_data = restrict_vdata;

   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          stride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *R_data_box;
   hypre_Box              *r_data_box;
   hypre_Box              *rc_data_box;
                       
   int                     Ri;
   int                     ri;
   int                     rci;
                         
   double                 *Rp0, *Rp1;
   double                 *rp, *rp0, *rp1;
   double                 *rcp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_Index             startc;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   int                     compute_i, i, j;
   int                     loopi, loopj, loopk;

   hypre_BeginTiming(restrict_data -> time_index);

   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   compute_pkg   = (restrict_data -> compute_pkg);
   cindex        = (restrict_data -> cindex);
   stride        = (restrict_data -> stride);

   stencil       = hypre_StructMatrixStencil(R);
   stencil_shape = hypre_StructStencilShape(stencil);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Restrict the residual.
    *--------------------------------------------------------------------*/

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

      hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            R_data_box  = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), i);
            r_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);
            rc_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(rc), i);

            Rp0 = hypre_StructMatrixBoxData(R, i, 0);
            Rp1 = hypre_StructMatrixBoxData(R, i, 1);
            rp  = hypre_StructVectorBoxData(r, i);
            rp0 = rp + hypre_BoxOffsetDistance(r_data_box, stencil_shape[0]);
            rp1 = rp + hypre_BoxOffsetDistance(r_data_box, stencil_shape[1]);
            rcp = hypre_StructVectorBoxData(rc, i);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);
                  hypre_SMGMapFineToCoarse(start, startc, cindex, stride);

                  hypre_GetStrideBoxSize(compute_box, stride, loop_size);
                  hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
                                 R_data_box,  startc, stridec, Ri,
                                 r_data_box,  start,  stride,  ri,
                                 rc_data_box, startc, stridec, rci,
                                 {
                                    rcp[rci] = rp[ri] + (Rp0[Ri] * rp0[ri] +
                                                         Rp1[Ri] * rp1[ri]);
                                 });
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
 * hypre_SMGRestrictFinalize
 *--------------------------------------------------------------------------*/

int
hypre_SMGRestrictFinalize( void *restrict_vdata )
{
   int ierr = 0;

   hypre_SMGRestrictData *restrict_data = restrict_vdata;

   if (restrict_data)
   {
      hypre_FreeComputePkg(restrict_data -> compute_pkg);
      hypre_FinalizeTiming(restrict_data -> time_index);
      hypre_TFree(restrict_data);
   }

   return ierr;
}


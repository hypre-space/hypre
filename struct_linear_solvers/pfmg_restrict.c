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

#include "headers.h"
#include "pfmg.h"


/*--------------------------------------------------------------------------
 * hypre_PFMGRestrictData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *RT;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         stride;

   int                 time_index;

} hypre_PFMGRestrictData;

/*--------------------------------------------------------------------------
 * hypre_PFMGRestrictCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PFMGRestrictCreate( )
{
   hypre_PFMGRestrictData *restrict_data;

   restrict_data = hypre_CTAlloc(hypre_PFMGRestrictData, 1);

   (restrict_data -> time_index)  = hypre_InitializeTiming("PFMGRestrict");
   
   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGRestrictSetup
 *--------------------------------------------------------------------------*/

int
hypre_PFMGRestrictSetup( void               *restrict_vdata,
                         hypre_StructMatrix *RT,
                         hypre_StructVector *r,
                         hypre_StructVector *rc,
                         hypre_Index         cindex,
                         hypre_Index         findex,
                         hypre_Index         stride          )
{
   hypre_PFMGRestrictData  *restrict_data = restrict_vdata;

   hypre_StructGrid        *grid;
   hypre_StructStencil     *stencil;
                          
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_processes;
   int                    **recv_processes;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;
                          
   hypre_ComputePkg        *compute_pkg;
                          
   int                      ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(r);
   stencil = hypre_StructMatrixStencil(RT);

   hypre_CreateComputeInfo(grid, stencil,
                        &send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes);

   hypre_ProjectBoxArrayArray(send_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(recv_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(indt_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(dept_boxes, cindex, stride);

   hypre_ComputePkgCreate(send_boxes, recv_boxes,
                          stride, stride,
                          send_processes, recv_processes,
                          indt_boxes, dept_boxes,
                          stride, grid,
                          hypre_StructVectorDataSpace(r), 1,
                          &compute_pkg);

   /*----------------------------------------------------------
    * Set up the restrict data structure
    *----------------------------------------------------------*/

   (restrict_data -> RT)          = hypre_StructMatrixRef(RT);
   (restrict_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex ,(restrict_data -> cindex));
   hypre_CopyIndex(stride ,(restrict_data -> stride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGRestrict:
 *--------------------------------------------------------------------------*/

int
hypre_PFMGRestrict( void               *restrict_vdata,
                    hypre_StructMatrix *RT,
                    hypre_StructVector *r,
                    hypre_StructVector *rc             )
{
   int ierr = 0;

   hypre_PFMGRestrictData  *restrict_data = restrict_vdata;

   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          stride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *RT_data_box;
   hypre_Box              *r_data_box;
   hypre_Box              *rc_data_box;
                       
   int                     RTi;
   int                     ri;
   int                     rci;
                         
   double                 *RTp0, *RTp1;
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

   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   hypre_BeginTiming(restrict_data -> time_index);

   compute_pkg   = (restrict_data -> compute_pkg);
   cindex        = (restrict_data -> cindex);
   stride        = (restrict_data -> stride);

   stencil       = hypre_StructMatrixStencil(RT);
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

            RT_data_box =
               hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RT), i);
            r_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);
            rc_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(rc), i);

            RTp0 = hypre_StructMatrixBoxData(RT, i, 1) -
               hypre_BoxOffsetDistance(RT_data_box, stencil_shape[1]);
            RTp1 = hypre_StructMatrixBoxData(RT, i, 0);
            rp  = hypre_StructVectorBoxData(r, i);
            rp0 = rp + hypre_BoxOffsetDistance(r_data_box, stencil_shape[0]);
            rp1 = rp + hypre_BoxOffsetDistance(r_data_box, stencil_shape[1]);
            rcp = hypre_StructVectorBoxData(rc, i);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);
                  hypre_PFMGMapFineToCoarse(start, cindex, stride, startc);

                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                  hypre_BoxLoop3Begin(loop_size,
                                      RT_data_box, startc, stridec,RTi,
                                      r_data_box, start, stride, ri,
                                      rc_data_box, startc, stridec, rci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,RTi,ri,rci
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop3For(loopi, loopj, loopk, RTi, ri, rci)
                     {
                        rcp[rci] = rp[ri] + (RTp0[RTi] * rp0[ri] +
                                             RTp1[RTi] * rp1[ri]);
                     }
                  hypre_BoxLoop3End(RTi, ri, rci);
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
 * hypre_PFMGRestrictDestroy
 *--------------------------------------------------------------------------*/

int
hypre_PFMGRestrictDestroy( void *restrict_vdata )
{
   int ierr = 0;

   hypre_PFMGRestrictData *restrict_data = restrict_vdata;

   if (restrict_data)
   {
      hypre_StructMatrixDestroy(restrict_data -> RT);
      hypre_ComputePkgDestroy(restrict_data -> compute_pkg);
      hypre_FinalizeTiming(restrict_data -> time_index);
      hypre_TFree(restrict_data);
   }

   return ierr;
}


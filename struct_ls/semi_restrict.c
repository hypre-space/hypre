/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
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

/*--------------------------------------------------------------------------
 * hypre_SemiRestrictData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *R;
   int                 R_stored_as_transpose;
   hypre_ComputePkg   *compute_pkg;
   hypre_Index         cindex;
   hypre_Index         stride;

   int                 time_index;

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

int
hypre_SemiRestrictSetup( void               *restrict_vdata,
                         hypre_StructMatrix *R,
                         int                 R_stored_as_transpose,
                         hypre_StructVector *r,
                         hypre_StructVector *rc,
                         hypre_Index         cindex,
                         hypre_Index         findex,
                         hypre_Index         stride                )
{
   hypre_SemiRestrictData *restrict_data = restrict_vdata;

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

int
hypre_SemiRestrict( void               *restrict_vdata,
                    hypre_StructMatrix *R,
                    hypre_StructVector *r,
                    hypre_StructVector *rc             )
{
   int ierr = 0;

   hypre_SemiRestrictData *restrict_data = restrict_vdata;

   int                     R_stored_as_transpose;
   hypre_ComputePkg       *compute_pkg;
   hypre_IndexRef          cindex;
   hypre_IndexRef          stride;

   hypre_StructGrid       *fgrid;
   int                    *fgrid_ids;
   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   int                    *cgrid_ids;

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *R_dbox;
   hypre_Box              *r_dbox;
   hypre_Box              *rc_dbox;
                       
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

   int                     compute_i, fi, ci, j;
   int                     loopi, loopj, loopk;

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
      hypre_ForBoxArrayI(ci, cgrid_boxes)
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
               Rp0 = hypre_StructMatrixBoxData(R, fi, 1) -
                  hypre_BoxOffsetDistance(R_dbox, stencil_shape[1]);
               Rp1 = hypre_StructMatrixBoxData(R, fi, 0);
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

int
hypre_SemiRestrictDestroy( void *restrict_vdata )
{
   int ierr = 0;

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


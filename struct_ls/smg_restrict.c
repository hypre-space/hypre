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
   hypre_Index         cstride;

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
                        hypre_Index         cstride,
                        hypre_Index         findex,
                        hypre_Index         fstride            )
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
                       
   hypre_SBoxArrayArray   *send_sboxes;
   hypre_SBoxArrayArray   *recv_sboxes;
   hypre_SBoxArrayArray   *indt_sboxes;
   hypre_SBoxArrayArray   *dept_sboxes;
                       
   hypre_ComputePkg       *compute_pkg;

   int                     ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(r);
   stencil = hypre_StructMatrixStencil(R);

   hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes,
                        grid, stencil);

   send_sboxes = hypre_ProjectBoxArrayArray(send_boxes, findex, fstride);
   recv_sboxes = hypre_ProjectBoxArrayArray(recv_boxes, findex, fstride);
   indt_sboxes = hypre_ProjectBoxArrayArray(indt_boxes, cindex, cstride);
   dept_sboxes = hypre_ProjectBoxArrayArray(dept_boxes, cindex, cstride);

   hypre_FreeBoxArrayArray(send_boxes);
   hypre_FreeBoxArrayArray(recv_boxes);
   hypre_FreeBoxArrayArray(indt_boxes);
   hypre_FreeBoxArrayArray(dept_boxes);

   compute_pkg = hypre_NewComputePkg(send_sboxes, recv_sboxes,
                                     send_processes, recv_processes,
                                     indt_sboxes, dept_sboxes,
                                     grid, hypre_StructVectorDataSpace(r), 1);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (restrict_data -> R)           = R;
   (restrict_data -> compute_pkg) = compute_pkg;
   hypre_CopyIndex(cindex ,(restrict_data -> cindex));
   hypre_CopyIndex(cstride ,(restrict_data -> cstride));

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
   hypre_IndexRef          cstride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_SBoxArrayArray   *compute_sbox_aa;
   hypre_SBoxArray        *compute_sbox_a;
   hypre_SBox             *compute_sbox;
                       
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
   hypre_IndexRef          stride;
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
   cstride       = (restrict_data -> cstride);

   stencil       = hypre_StructMatrixStencil(R);
   stencil_shape = hypre_StructStencilShape(stencil);

   stride = cstride;
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
            comm_handle = hypre_InitializeIndtComputations(compute_pkg, rp);
            compute_sbox_aa = hypre_ComputePkgIndtSBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            hypre_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = hypre_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      hypre_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

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

            hypre_ForSBoxI(j, compute_sbox_a)
               {
                  compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

                  start  = hypre_SBoxIMin(compute_sbox);
                  hypre_SMGMapFineToCoarse(start, startc, cindex, cstride);

                  hypre_GetSBoxSize(compute_sbox, loop_size);
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


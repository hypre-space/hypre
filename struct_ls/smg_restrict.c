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
 * zzz_SMGRestrictData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructMatrix *R;
   zzz_ComputePkg   *compute_pkg;
   zzz_Index         cindex;
   zzz_Index         cstride;

   int               time_index;

} zzz_SMGRestrictData;

/*--------------------------------------------------------------------------
 * zzz_SMGRestrictInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGRestrictInitialize( )
{
   zzz_SMGRestrictData *restrict_data;

   restrict_data = zzz_CTAlloc(zzz_SMGRestrictData, 1);
   (restrict_data -> time_index)  = zzz_InitializeTiming("SMGRestrict");

   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRestrictSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGRestrictSetup( void             *restrict_vdata,
                      zzz_StructMatrix *R,
                      zzz_StructVector *r,
                      zzz_StructVector *rc,
                      zzz_Index         cindex,
                      zzz_Index         cstride,
                      zzz_Index         findex,
                      zzz_Index         fstride            )
{
   zzz_SMGRestrictData  *restrict_data = restrict_vdata;

   zzz_StructGrid       *grid;
   zzz_StructStencil    *stencil;
                       
   zzz_BoxArrayArray    *send_boxes;
   zzz_BoxArrayArray    *recv_boxes;
   int                 **send_box_ranks;
   int                 **recv_box_ranks;
   zzz_BoxArrayArray    *indt_boxes;
   zzz_BoxArrayArray    *dept_boxes;
                       
   zzz_SBoxArrayArray   *send_sboxes;
   zzz_SBoxArrayArray   *recv_sboxes;
   zzz_SBoxArrayArray   *indt_sboxes;
   zzz_SBoxArrayArray   *dept_sboxes;
                       
   zzz_ComputePkg       *compute_pkg;

   int                   ierr;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = zzz_StructVectorGrid(r);
   stencil = zzz_StructMatrixStencil(R);

   zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, findex, fstride);
   recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, findex, fstride);
   indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, cindex, cstride);
   dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, cindex, cstride);

   zzz_FreeBoxArrayArray(send_boxes);
   zzz_FreeBoxArrayArray(recv_boxes);
   zzz_FreeBoxArrayArray(indt_boxes);
   zzz_FreeBoxArrayArray(dept_boxes);

   compute_pkg = zzz_NewComputePkg(send_sboxes, recv_sboxes,
                                   send_box_ranks, recv_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, zzz_StructVectorDataSpace(r), 1);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (restrict_data -> R)           = R;
   (restrict_data -> compute_pkg) = compute_pkg;
   zzz_CopyIndex(cindex ,(restrict_data -> cindex));
   zzz_CopyIndex(cstride ,(restrict_data -> cstride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRestrict:
 *
 * Notes:
 *    This routine assumes that the interpolation stencil is a
 *    2-point stencil.
 *--------------------------------------------------------------------------*/

int
zzz_SMGRestrict( void             *restrict_vdata,
                 zzz_StructMatrix *R,
                 zzz_StructVector *r,
                 zzz_StructVector *rc             )
{
   int ierr;

   zzz_SMGRestrictData  *restrict_data = restrict_vdata;

   zzz_ComputePkg       *compute_pkg;
   zzz_IndexRef          cindex;
   zzz_IndexRef          cstride;

   zzz_CommHandle       *comm_handle;
                       
   zzz_SBoxArrayArray   *compute_sbox_aa;
   zzz_SBoxArray        *compute_sbox_a;
   zzz_SBox             *compute_sbox;
                       
   zzz_Box              *R_data_box;
   zzz_Box              *r_data_box;
   zzz_Box              *rc_data_box;
                       
   int                   Ri;
   int                   ri;
   int                   rci;
                       
   double               *Rp0, *Rp1;
   double               *rp, *rp0, *rp1;
   double               *rcp;
                       
   zzz_Index             loop_size;
   zzz_IndexRef          start;
   zzz_IndexRef          stride;
   zzz_Index             startc;
   zzz_Index             stridec;
                       
   zzz_StructStencil    *stencil;
   zzz_Index            *stencil_shape;

   int                   compute_i, i, j;
   int                   loopi, loopj, loopk;

   zzz_BeginTiming(restrict_data -> time_index);

   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   compute_pkg   = (restrict_data -> compute_pkg);
   cindex        = (restrict_data -> cindex);
   cstride       = (restrict_data -> cstride);

   stencil       = zzz_StructMatrixStencil(R);
   stencil_shape = zzz_StructStencilShape(stencil);

   stride = cstride;
   zzz_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Restrict the residual.
    *--------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            rp = zzz_StructVectorData(r);
            comm_handle = zzz_InitializeIndtComputations(compute_pkg, rp);
            compute_sbox_aa = zzz_ComputePkgIndtSBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            zzz_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = zzz_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      zzz_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         R_data_box  = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(R), i);
         r_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(r), i);
         rc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(rc), i);

         Rp0 = zzz_StructMatrixBoxData(R, i, 0);
         Rp1 = zzz_StructMatrixBoxData(R, i, 1);
         rp  = zzz_StructVectorBoxData(r, i);
         rp0 = rp + zzz_BoxOffsetDistance(r_data_box, stencil_shape[0]);
         rp1 = rp + zzz_BoxOffsetDistance(r_data_box, stencil_shape[1]);
         rcp = zzz_StructVectorBoxData(rc, i);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            start  = zzz_SBoxIMin(compute_sbox);
            zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

            zzz_GetSBoxSize(compute_sbox, loop_size);
            zzz_BoxLoop3(loopi, loopj, loopk, loop_size,
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

   zzz_IncFLOPCount(4*zzz_StructVectorGlobalSize(rc));
   zzz_EndTiming(restrict_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRestrictFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGRestrictFinalize( void *restrict_vdata )
{
   int ierr;

   zzz_SMGRestrictData *restrict_data = restrict_vdata;

   if (restrict_data)
   {
      zzz_FreeComputePkg(restrict_data -> compute_pkg);
      zzz_FinalizeTiming(restrict_data -> time_index);
      zzz_TFree(restrict_data);
   }

   return ierr;
}


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
   zzz_SBoxArray        *coarse_points;
   zzz_Index            *cindex;
   zzz_Index            *cstride;

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
                       
   double               *Rp0, Rp1;
   double               *rp, rp0, rp1;
   double               *rcp;
                       
   zzz_Box              *box;
   zzz_Index            *index;
   zzz_Index            *start;
   zzz_Index            *stride;
   zzz_Index            *startc;
   zzz_Index            *stridec;
                       
   zzz_StructStencil    *stencil;
   zzz_Index           **stencil_shape;

   int                   compute_i, i, j, d;

   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   compute_pkg   = (restrict_data -> compute_pkg);
   coarse_points = (restrict_data -> coarse_points);
   cindex        = (restrict_data -> cindex);
   cstride       = (restrict_data -> cstride);

   stencil       = zzz_StructMatrixStencil(R);
   stencil_shape = zzz_StructStencilShape(stencil);

   index = zzz_NewIndex();

   startc = zzz_NewIndex();

   stride = cstride;
   stridec = zzz_NewIndex();
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

            box    = zzz_SBoxBox(compute_sbox);

            start  = zzz_SBoxIMin(compute_sbox);
            zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

            zzz_BoxLoop3(box, index,
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

   zzz_FreeIndex(index);
   zzz_FreeIndex(startc);
   zzz_FreeIndex(stridec);

   return ierr;
}


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
 * zzz_SMGIntAdd:
 *
 * Notes:
 *    This routine assumes that the interpolation stencil is a
 *    2-point stencil and that stencil_shape[0] = -stencil_shape[1].
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAdd( void             *intadd_vdata,
               zzz_StructMatrix *PT,
               zzz_StructVector *xc,
               zzz_StructVector *x           )
{
   int ierr;

   zzz_SMGIntAddData    *intadd_data = intadd_vdata;

   zzz_ComputePkg       *compute_pkg;
   zzz_SBoxArray        *coarse_points;
   zzz_Index            *cindex;
   zzz_Index            *cstride;

   zzz_CommHandle       *comm_handle;
                       
   zzz_SBoxArrayArray   *compute_sbox_aa;
   zzz_SBoxArray        *compute_sbox_a;
   zzz_SBox             *compute_sbox;
                       
   zzz_Box              *PT_data_box;
   zzz_Box              *xc_data_box;
   zzz_Box              *x_data_box;
                       
   int                   Pi;
   int                   xci;
   int                   xi;
                       
   double               *Pp0, *Pp1;
   double               *xcp, *xcp0, *xcp1;
   double               *xp;
                       
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
    * Initialize some things
    *-----------------------------------------------------------------------*/

   compute_pkg   = (intadd_data -> compute_pkg);
   coarse_points = (intadd_data -> coarse_points);
   cindex        = (intadd_data -> cindex);
   cstride       = (intadd_data -> cstride);

   stencil       = zzz_StructMatrixStencil(PT);
   stencil_shape = zzz_StructStencilShape(stencil);

   index = zzz_NewIndex();

   startc = zzz_NewIndex();

   stride = cstride;
   stridec = zzz_NewIndex();
   zzz_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Interpolate the coarse error and add to the fine grid solution.
    *--------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xcp = zzz_StructVectorData(xc);
            comm_handle = zzz_InitializeIndtComputations(compute_pkg, xcp);
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

      /*----------------------------------------
       * Compute x at coarse points.
       *----------------------------------------*/

      if (compute_i == 0)
      {
         compute_sbox_a = coarse_points;
         zzz_ForSBoxI(i, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);

            box = zzz_SBoxBox(compute_sbox);

            start   = zzz_SBoxIMin(compute_sbox);
            zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

            xc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(xc), i);
            x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);

            xcp = zzz_StructVectorBoxData(xc, i);
            xp  = zzz_StructVectorBoxData(x, i);

            zzz_BoxLoop2(box, index,
                         xc_data_box, startc, stridec, xci,
                         x_data_box,  start,  stride,  xi,
                         {
                            xp[xi] += xcp[xci];
                         });
         }
      }

      /*----------------------------------------
       * Compute x at fine points.
       *
       * Loop over fine points and associate
       * for each fine point the coarse point
       * in the direction of -stencil_shape[1].
       *----------------------------------------*/

      zzz_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
         xc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(xc), i);
         x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);

         Pp0  = zzz_StructMatrixBoxData(PT, i, 1);
         Pp1  = zzz_StructMatrixBoxData(PT, i, 0) +
            zzz_BoxOffsetDistance(PT_data_box, stencil_shape[1]);
         xcp0 = zzz_StructVectorBoxData(xc, i);
         xcp1 = xcp0 + zzz_BoxOffsetDistance(xc_data_box, stencil_shape[1]);
         xp   = zzz_StructVectorBoxData(x, i);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            box    = zzz_SBoxBox(compute_sbox);

            start  = zzz_SBoxIMin(compute_sbox);
            for (d = 0; d < 3; d++)
            {
               zzz_IndexD(startc, d) =
                  zzz_IndexD(start, d) - zzz_IndexD(stencil_shape[1], d);
            }
            zzz_SMGMapFineToCoarse(startc, startc, cindex, cstride);

            zzz_BoxLoop3(box, index,
                         PT_data_box, startc, stridec, Pi,
                         xc_data_box, startc, stridec, xci,
                         x_data_box,  start,  stride,  xi,
                         {
                            xp[xi] += (Pp0[Pi] * xcp0[xci] +
                                       Pp1[Pi] * xcp1[xci]);
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


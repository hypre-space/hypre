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
 * zzz_SMGIntAddData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructMatrix *PT;
   zzz_ComputePkg   *compute_pkg;
   zzz_SBoxArray    *coarse_points;
   zzz_Index        *cindex;
   zzz_Index        *cstride;

} zzz_SMGIntAddData;

/*--------------------------------------------------------------------------
 * zzz_SMGIntAddInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGIntAddInitialize( )
{
   zzz_SMGIntAddData *intadd_data;

   intadd_data = zzz_CTAlloc(zzz_SMGIntAddData, 1);

   return (void *) intadd_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGIntAddSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAddSetup( void             *intadd_vdata,
                    zzz_StructMatrix *PT,
                    zzz_StructVector *xc,
                    zzz_StructVector *x,
                    zzz_Index        *cindex,
                    zzz_Index        *cstride,
                    zzz_Index        *findex,
                    zzz_Index        *fstride          )
{
   zzz_SMGIntAddData    *intadd_data = intadd_vdata;

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
                       
   zzz_SBoxArrayArray   *sbox_array_array;
   zzz_SBoxArray        *sbox_array;
   zzz_SBox             *sbox;

   zzz_ComputePkg       *compute_pkg;
   zzz_SBoxArray        *coarse_points;

   int                   i, j, k;
   int                   ierr;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = zzz_StructVectorGrid(x);
   stencil = zzz_StructMatrixStencil(PT);

   zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, cindex, cstride);
   recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, cindex, cstride);
   indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, findex, fstride);
   dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, findex, fstride);

   zzz_FreeBoxArrayArray(send_boxes);
   zzz_FreeBoxArrayArray(recv_boxes);
   zzz_FreeBoxArrayArray(indt_boxes);
   zzz_FreeBoxArrayArray(dept_boxes);

   /* Do communications on the coarse grid */
   for (k = 0; k < 2; k++)
   {
      switch (k)
      {
         case 0:
         sbox_array_array = send_sboxes;
         break;

         case 1:
         sbox_array_array = recv_sboxes;
         break;
      }

      zzz_ForSBoxArrayI(i, send_sboxes)
      {
         sbox_array = zzz_SBoxArrayArraySBoxArray(send_sboxes, i);
         zzz_ForSBoxArrayI(j, sbox_array)
         {
            sbox = zzz_SBoxArraySBox(sbox_array, j);
            zzz_SMGMapFineToCoarse(zzz_SBoxIMin(sbox), zzz_SBoxIMin(sbox),
                                   cindex, cstride);
            zzz_SMGMapFineToCoarse(zzz_SBoxIMax(sbox), zzz_SBoxIMax(sbox),
                                   cindex, cstride);
            zzz_SetIndex(zzz_SBoxStride(sbox), 1, 1, 1);
         }
      }
   }

   compute_pkg = zzz_NewComputePkg(send_sboxes, recv_sboxes,
                                   send_box_ranks, recv_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, zzz_StructVectorDataSpace(xc), 1);

   /*----------------------------------------------------------
    * Set up the coarse points SBoxArray
    *----------------------------------------------------------*/

   coarse_points = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid),
                                       cindex, cstride);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (intadd_data -> PT)            = PT;
   (intadd_data -> compute_pkg)   = compute_pkg;
   (intadd_data -> coarse_points) = coarse_points;
   (intadd_data -> cindex)        = cindex;
   (intadd_data -> cstride)       = cstride;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGIntAdd:
 *
 * Notes:
 *    This routine assumes that the interpolation stencil is a
 *    2-point stencil and that stencil_shape[0] = -stencil_shape[1].
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAdd( void             *intadd_vdata,
               zzz_StructVector *xc,
               zzz_StructVector *x           )
{
   int ierr;

   zzz_SMGIntAddData    *intadd_data = intadd_vdata;

   zzz_StructMatrix     *PT = (intadd_data -> PT);

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
                       
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;
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

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();

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

            start = zzz_SBoxIMin(compute_sbox);
            zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

            xc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(xc), i);
            x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);

            xcp = zzz_StructVectorBoxData(xc, i);
            xp  = zzz_StructVectorBoxData(x, i);

            zzz_GetSBoxSize(compute_sbox, loop_size);
            zzz_BoxLoop2(loop_index, loop_size,
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

            start  = zzz_SBoxIMin(compute_sbox);
            for (d = 0; d < 3; d++)
            {
               zzz_IndexD(startc, d) =
                  zzz_IndexD(start, d) - zzz_IndexD(stencil_shape[1], d);
            }
            zzz_SMGMapFineToCoarse(startc, startc, cindex, cstride);

            zzz_GetSBoxSize(compute_sbox, loop_size);
            zzz_BoxLoop3(loop_index, loop_size,
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

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(startc);
   zzz_FreeIndex(stridec);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGIntAddFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAddFinalize( void *intadd_vdata )
{
   int ierr;

   zzz_SMGIntAddData *intadd_data = intadd_vdata;

   if (intadd_data)
   {
      zzz_FreeSBoxArray(intadd_data -> coarse_points);
      zzz_FreeComputePkg(intadd_data -> compute_pkg);
      zzz_TFree(intadd_data);
   }

   return ierr;
}


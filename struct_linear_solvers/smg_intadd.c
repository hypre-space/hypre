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

   int               time_index;

} zzz_SMGIntAddData;

/*--------------------------------------------------------------------------
 * zzz_SMGIntAddInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGIntAddInitialize( )
{
   zzz_SMGIntAddData *intadd_data;

   intadd_data = zzz_CTAlloc(zzz_SMGIntAddData, 1);
   (intadd_data -> time_index)  = zzz_InitializeTiming("SMGIntAdd");

   return (void *) intadd_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGIntAddSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAddSetup( void             *intadd_vdata,
                    zzz_StructMatrix *PT,
                    zzz_StructVector *xc,
                    zzz_StructVector *e,
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
                       
   zzz_ComputePkg       *compute_pkg;
   zzz_SBoxArray        *coarse_points;

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

   send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, findex, fstride);
   recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, findex, fstride);
   indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, findex, fstride);
   dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, findex, fstride);

   zzz_FreeBoxArrayArray(send_boxes);
   zzz_FreeBoxArrayArray(recv_boxes);
   zzz_FreeBoxArrayArray(indt_boxes);
   zzz_FreeBoxArrayArray(dept_boxes);

   /* reverse send and recv info */
   compute_pkg = zzz_NewComputePkg(recv_sboxes, send_sboxes,
                                   recv_box_ranks, send_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, zzz_StructVectorDataSpace(e), 1);

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
 *
 * This routine assumes that the interpolation stencil is a
 * 2-point stencil and that stencil_shape[0] = -stencil_shape[1].
 *
 * This is a complex algorithm that proceeds as follows:
 *   1) Compute the result of applying the off-diagonal coefficients
 *      of P^T to the coarse solution vector, storing the results in
 *      the fine vector, `e'.  The two stencil coefficient products
 *      are interleaved so that: 
 *         - the stencil-element-0 product lives at C-points
 *         - the stencil-element-1 product lives at F-points.
 *   2) Communicate boundary information for e while computing
 *      the updates for the fine solution vector.  The communication
 *      is done "backwards" so that ghost data gets copied into real
 *      grid data locations.
 *--------------------------------------------------------------------------*/

int
zzz_SMGIntAdd( void             *intadd_vdata,
               zzz_StructMatrix *PT,
               zzz_StructVector *xc,
               zzz_StructVector *e,
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
   zzz_Box              *e_data_box;
                       
   int                   PTi;
   int                   xci;
   int                   xi;
   int                   ei;
                       
   double               *xcp;
   double               *PTp0, *PTp1;
   double               *ep, *ep0, *ep1;
   double               *xp;
                       
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;
   zzz_Index            *start;
   zzz_Index            *stride;
   zzz_Index            *startc;
   zzz_Index            *stridec;
                       
   zzz_StructStencil    *stencil;
   zzz_Index           **stencil_shape;

   int                   compute_i, i, j;

   zzz_BeginTiming(intadd_data -> time_index);

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
    * Compute e = (P^T)_off x_c, where (P^T)_off corresponds to the
    * off-diagonal coefficients of P^T.  Interleave the results.
    *--------------------------------------------------------------------*/

   compute_sbox_a = coarse_points;
   zzz_ForSBoxI(i, compute_sbox_a)
   {
      compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);

      start = zzz_SBoxIMin(compute_sbox);
      zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

      e_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(e), i);
      PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
      xc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(xc), i);

      ep0 = zzz_StructVectorBoxData(e, i);
      ep1 = zzz_StructVectorBoxData(e, i) +
         zzz_BoxOffsetDistance(e_data_box, stencil_shape[1]);
      PTp0 = zzz_StructMatrixBoxData(PT, i, 0);
      PTp1 = zzz_StructMatrixBoxData(PT, i, 1);
      xcp = zzz_StructVectorBoxData(xc, i);

      zzz_GetSBoxSize(compute_sbox, loop_size);
      zzz_BoxLoop3(loop_index, loop_size,
                   e_data_box,  start,  stride,  ei,
                   PT_data_box, startc, stridec, PTi,
                   xc_data_box, startc, stridec, xci,
                   {
                      ep0[ei] = PTp0[PTi]*xcp[xci];
                      ep1[ei] = PTp1[PTi]*xcp[xci];
                   });
   }

   /*--------------------------------------------------------------------
    * Interpolate the coarse error and add to the fine grid solution.
    *--------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            ep = zzz_StructVectorData(e);
            comm_handle = zzz_InitializeIndtComputations(compute_pkg, ep);
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

            x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
            xc_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(xc), i);

            xp = zzz_StructVectorBoxData(x, i);
            xcp = zzz_StructVectorBoxData(xc, i);

            zzz_GetSBoxSize(compute_sbox, loop_size);
            zzz_BoxLoop2(loop_index, loop_size,
                         x_data_box,  start,  stride,  xi,
                         xc_data_box, startc, stridec, xci,
                         {
                            xp[xi] += xcp[xci];
                         });
         }
      }

      /*----------------------------------------
       * Compute x at fine points.
       *----------------------------------------*/

      zzz_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
         e_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(e), i);

         xp  = zzz_StructVectorBoxData(x, i);
         ep0 = zzz_StructVectorBoxData(e, i);
         ep1 = zzz_StructVectorBoxData(e, i) +
            zzz_BoxOffsetDistance(e_data_box, stencil_shape[1]);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            start  = zzz_SBoxIMin(compute_sbox);

            zzz_GetSBoxSize(compute_sbox, loop_size);
            zzz_BoxLoop2(loop_index, loop_size,
                         x_data_box, start, stride, xi,
                         e_data_box, start, stride, ei,
                         {
                            xp[xi] += ep0[ei] + ep1[ei];
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

   zzz_IncFLOPCount(5*zzz_StructVectorGlobalSize(xc));
   zzz_EndTiming(intadd_data -> time_index);

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
      zzz_FinalizeTiming(intadd_data -> time_index);
      zzz_TFree(intadd_data);
   }

   return ierr;
}


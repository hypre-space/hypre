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
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_StructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructMatrix  *A;
   zzz_StructVector  *x;
   zzz_ComputePkg    *compute_pkg;

} zzz_StructMatvecData;

/*--------------------------------------------------------------------------
 * zzz_StructMatvecInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_StructMatvecInitialize( )
{
   zzz_StructMatvecData *matvec_data;

   matvec_data = zzz_CTAlloc(zzz_StructMatvecData, 1);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * zzz_StructMatvecSetup
 *--------------------------------------------------------------------------*/

int
zzz_StructMatvecSetup( void             *matvec_vdata,
                       zzz_StructMatrix *A,
                       zzz_StructVector *x            )
{
   int ierr;

   zzz_StructMatvecData *matvec_data = matvec_vdata;

   zzz_StructGrid       *grid;
   zzz_StructStencil    *stencil;
                       
   zzz_BoxArrayArray    *send_boxes;
   zzz_BoxArrayArray    *recv_boxes;
   int                 **send_box_ranks;
   int                 **recv_box_ranks;
   zzz_BoxArrayArray    *indt_boxes;
   zzz_BoxArrayArray    *dept_boxes;
                       
   zzz_SBoxArrayArray    *send_sboxes;
   zzz_SBoxArrayArray    *recv_sboxes;
   zzz_SBoxArrayArray    *indt_sboxes;
   zzz_SBoxArrayArray    *dept_sboxes;
                       
   zzz_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = zzz_StructMatrixGrid(A);
   stencil = zzz_StructMatrixStencil(A);

   zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = zzz_ConvertToSBoxArrayArray(send_boxes);
   recv_sboxes = zzz_ConvertToSBoxArrayArray(recv_boxes);
   indt_sboxes = zzz_ConvertToSBoxArrayArray(indt_boxes);
   dept_sboxes = zzz_ConvertToSBoxArrayArray(dept_boxes);

   compute_pkg = zzz_NewComputePkg(send_sboxes, recv_sboxes,
                                   send_box_ranks, recv_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, zzz_StructVectorDataSpace(x), 1);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = A;
   (matvec_data -> x)           = x;
   (matvec_data -> compute_pkg) = compute_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_StructMatvecCompute
 *--------------------------------------------------------------------------*/

int
zzz_StructMatvecCompute( void             *matvec_vdata,
                         double            alpha,
                         double            beta,
                         zzz_StructVector *y            )
{
   int ierr;

   zzz_StructMatvecData *matvec_data = matvec_vdata;

   zzz_StructMatrix     *A;
   zzz_StructVector     *x;
   zzz_ComputePkg       *compute_pkg;

   zzz_CommHandle       *comm_handle;
                       
   zzz_SBoxArrayArray   *compute_sbox_aa;
   zzz_SBoxArray        *compute_sbox_a;
   zzz_SBox             *compute_sbox;
                       
   zzz_Box              *A_data_box;
   zzz_Box              *x_data_box;
   zzz_Box              *y_data_box;
                       
   int                   Ai;
   int                   xi, xoffset;
   int                   yi;
                       
   double               *Ap;
   double               *xp;
   double               *yp;
                       
   zzz_BoxArray         *boxes;
   zzz_Box              *box;
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;
   zzz_Index            *start;
   zzz_Index            *stride;
   zzz_Index            *unit_stride;
                       
   zzz_StructStencil    *stencil;
   zzz_Index           **stencil_shape;
   int                   stencil_size;

   double                temp;
   int                   compute_i, i, j, si;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   A           = (matvec_data -> A);
   x           = (matvec_data -> x);
   compute_pkg = (matvec_data -> compute_pkg);

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();

   unit_stride = zzz_NewIndex();
   zzz_SetIndex(unit_stride, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(A));
      zzz_ForBoxI(i, boxes)
      {
         box   = zzz_BoxArrayBox(boxes, i);
         start = zzz_BoxIMin(box);

         y_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);
         yp = zzz_StructVectorBoxData(y, i);

         zzz_GetBoxSize(box, loop_size);
         zzz_BoxLoop1(loop_index, loop_size,
                      y_data_box, start, unit_stride, yi,
                      {
                         yp[yi] *= beta;
                      });
      }

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * Do (alpha != 0.0) computation
    *-----------------------------------------------------------------------*/

   stencil       = zzz_StructMatrixStencil(A);
   stencil_shape = zzz_StructStencilShape(stencil);
   stencil_size  = zzz_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = zzz_StructVectorData(x);
            comm_handle = zzz_InitializeIndtComputations(compute_pkg, xp);
            compute_sbox_aa = zzz_ComputePkgIndtSBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y
             *--------------------------------------------------------------*/

            temp = beta / alpha;
            if (temp != 1.0)
            {
               boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(A));
               zzz_ForBoxI(i, boxes)
               {
                  box   = zzz_BoxArrayBox(boxes, i);
                  start = zzz_BoxIMin(box);

                  y_data_box =
                     zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);
                  yp = zzz_StructVectorBoxData(y, i);

                  if (temp == 0.0)
                  {
                     zzz_GetBoxSize(box, loop_size);
                     zzz_BoxLoop1(loop_index, loop_size,
                                  y_data_box, start, unit_stride, yi,
                                  {
                                     yp[yi] = 0.0;
                                  });
                  }
                  else
                  {
                     zzz_GetBoxSize(box, loop_size);
                     zzz_BoxLoop1(loop_index, loop_size,
                                  y_data_box, start, unit_stride, yi,
                                  {
                                     yp[yi] *= temp;
                                  });
                  }
               }
            }
         }
         break;

         case 1:
         {
            zzz_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = zzz_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * y += A*x
       *--------------------------------------------------------------------*/

      zzz_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
         x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
         y_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);

         xp = zzz_StructVectorBoxData(x, i);
         yp = zzz_StructVectorBoxData(y, i);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            box    = zzz_SBoxBox(compute_sbox);
            start  = zzz_SBoxIMin(compute_sbox);
            stride = zzz_SBoxStride(compute_sbox);

            for (si = 0; si < stencil_size; si++)
            {
               Ap = zzz_StructMatrixBoxData(A, i, si);

               xoffset = zzz_BoxOffsetDistance(x_data_box, stencil_shape[si]);

               zzz_GetBoxSize(box, loop_size);
               zzz_BoxLoop3(loop_index, loop_size,
                            A_data_box, start, stride, Ai,
                            x_data_box, start, stride, xi,
                            y_data_box, start, stride, yi,
                            {
                               yp[yi] += Ap[Ai] * xp[xi + xoffset];
                            });
            }

            if (alpha != 1.0)
            {
               zzz_GetBoxSize(box, loop_size);
               zzz_BoxLoop1(loop_index, loop_size,
                            y_data_box, start, stride, yi,
                            {
                               yp[yi] *= alpha;
                            });
            }
         }
      }
   }
   
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(unit_stride);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_StructMatvecFinalize
 *--------------------------------------------------------------------------*/

int
zzz_StructMatvecFinalize( void *matvec_vdata )
{
   int ierr;

   zzz_StructMatvecData *matvec_data = matvec_vdata;

   if (matvec_data)
   {
      zzz_FreeComputePkg(matvec_data -> compute_pkg );
      zzz_TFree(matvec_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_StructMatvec
 *--------------------------------------------------------------------------*/

int
zzz_StructMatvec( double            alpha,
                  zzz_StructMatrix *A,
                  zzz_StructVector *x,
                  double            beta,
                  zzz_StructVector *y     )
{
   int ierr;

   void *matvec_data;

   matvec_data = zzz_StructMatvecInitialize();
   ierr = zzz_StructMatvecSetup(matvec_data, A, x);
   ierr = zzz_StructMatvecCompute(matvec_data, alpha, beta, y);
   ierr = zzz_StructMatvecFinalize(matvec_data);

   return ierr;
}

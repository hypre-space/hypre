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
 * hypre_StructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix  *A;
   hypre_StructVector  *x;
   hypre_ComputePkg    *compute_pkg;

} hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_StructMatvecInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_StructMatvecInitialize( )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData, 1);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecSetup
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecSetup( void             *matvec_vdata,
                       hypre_StructMatrix *A,
                       hypre_StructVector *x            )
{
   int ierr;

   hypre_StructMatvecData *matvec_data = matvec_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                 **send_box_ranks;
   int                 **recv_box_ranks;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_SBoxArrayArray    *send_sboxes;
   hypre_SBoxArrayArray    *recv_sboxes;
   hypre_SBoxArrayArray    *indt_sboxes;
   hypre_SBoxArrayArray    *dept_sboxes;
                       
   hypre_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = hypre_ConvertToSBoxArrayArray(send_boxes);
   recv_sboxes = hypre_ConvertToSBoxArrayArray(recv_boxes);
   indt_sboxes = hypre_ConvertToSBoxArrayArray(indt_boxes);
   dept_sboxes = hypre_ConvertToSBoxArrayArray(dept_boxes);

   compute_pkg = hypre_NewComputePkg(send_sboxes, recv_sboxes,
                                   send_box_ranks, recv_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, hypre_StructVectorDataSpace(x), 1);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = A;
   (matvec_data -> x)           = x;
   (matvec_data -> compute_pkg) = compute_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecCompute( void             *matvec_vdata,
                         double            alpha,
                         double            beta,
                         hypre_StructVector *y            )
{
   int ierr;

   hypre_StructMatvecData *matvec_data = matvec_vdata;

   hypre_StructMatrix     *A;
   hypre_StructVector     *x;
   hypre_ComputePkg       *compute_pkg;

   hypre_CommHandle       *comm_handle;
                       
   hypre_SBoxArrayArray   *compute_sbox_aa;
   hypre_SBoxArray        *compute_sbox_a;
   hypre_SBox             *compute_sbox;
                       
   hypre_Box              *A_data_box;
   hypre_Box              *x_data_box;
   hypre_Box              *y_data_box;
                       
   int                   Ai;
   int                   xi, xoffset;
   int                   yi;
                       
   double               *Ap;
   double               *xp;
   double               *yp;
                       
   hypre_BoxArray         *boxes;
   hypre_Box              *box;
   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_IndexRef          stride;
   hypre_Index             unit_stride;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;
   int                   stencil_size;

   double                temp;
   int                   compute_i, i, j, si;
   int                   loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   A           = (matvec_data -> A);
   x           = (matvec_data -> x);
   compute_pkg = (matvec_data -> compute_pkg);

   hypre_SetIndex(unit_stride, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
      hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_GetBoxSize(box, loop_size);
         hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
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

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            comm_handle = hypre_InitializeIndtComputations(compute_pkg, xp);
            compute_sbox_aa = hypre_ComputePkgIndtSBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y
             *--------------------------------------------------------------*/

            temp = beta / alpha;
            if (temp != 1.0)
            {
               boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
               hypre_ForBoxI(i, boxes)
               {
                  box   = hypre_BoxArrayBox(boxes, i);
                  start = hypre_BoxIMin(box);

                  y_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  yp = hypre_StructVectorBoxData(y, i);

                  if (temp == 0.0)
                  {
                     hypre_GetBoxSize(box, loop_size);
                     hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
                                  y_data_box, start, unit_stride, yi,
                                  {
                                     yp[yi] = 0.0;
                                  });
                  }
                  else
                  {
                     hypre_GetBoxSize(box, loop_size);
                     hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
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
            hypre_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = hypre_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * y += A*x
       *--------------------------------------------------------------------*/

      hypre_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

            hypre_GetSBoxSize(compute_sbox, loop_size);
            start  = hypre_SBoxIMin(compute_sbox);
            stride = hypre_SBoxStride(compute_sbox);

            for (si = 0; si < stencil_size; si++)
            {
               Ap = hypre_StructMatrixBoxData(A, i, si);

               xoffset = hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

               hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
                            A_data_box, start, stride, Ai,
                            x_data_box, start, stride, xi,
                            y_data_box, start, stride, yi,
                            {
                               yp[yi] += Ap[Ai] * xp[xi + xoffset];
                            });
            }

            if (alpha != 1.0)
            {
               hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
                            y_data_box, start, stride, yi,
                            {
                               yp[yi] *= alpha;
                            });
            }
         }
      }
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecFinalize
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecFinalize( void *matvec_vdata )
{
   int ierr;

   hypre_StructMatvecData *matvec_data = matvec_vdata;

   if (matvec_data)
   {
      hypre_FreeComputePkg(matvec_data -> compute_pkg );
      hypre_TFree(matvec_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvec
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvec( double            alpha,
                  hypre_StructMatrix *A,
                  hypre_StructVector *x,
                  double            beta,
                  hypre_StructVector *y     )
{
   int ierr;

   void *matvec_data;

   matvec_data = hypre_StructMatvecInitialize();
   ierr = hypre_StructMatvecSetup(matvec_data, A, x);
   ierr = hypre_StructMatvecCompute(matvec_data, alpha, beta, y);
   ierr = hypre_StructMatvecFinalize(matvec_data);

   return ierr;
}

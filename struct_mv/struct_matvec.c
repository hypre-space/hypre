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

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

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
 * hypre_StructMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_StructMatvecCreate( )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData, 1);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecSetup
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecSetup( void               *matvec_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *x            )
{
   int ierr = 0;

   hypre_StructMatvecData  *matvec_data = matvec_vdata;
                          
   hypre_StructGrid        *grid;
   hypre_StructStencil     *stencil;
                          
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_processes;
   int                    **recv_processes;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;

   hypre_Index              unit_stride;
                       
   hypre_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   hypre_CreateComputeInfo(grid, stencil,
                        &send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes);

   hypre_SetIndex(unit_stride, 1, 1, 1);
   hypre_ComputePkgCreate(send_boxes, recv_boxes,
                          unit_stride, unit_stride,
                          send_processes, recv_processes,
                          indt_boxes, dept_boxes,
                          unit_stride,
                          grid, hypre_StructVectorDataSpace(x), 1,
                          &compute_pkg);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = hypre_StructMatrixRef(A);
   (matvec_data -> x)           = hypre_StructVectorRef(x);
   (matvec_data -> compute_pkg) = compute_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecCompute( void               *matvec_vdata,
                           double              alpha,
                           hypre_StructMatrix *A,
                           hypre_StructVector *x,
                           double              beta,
                           hypre_StructVector *y            )
{
   int ierr = 0;

   hypre_StructMatvecData  *matvec_data = matvec_vdata;
                          
   hypre_ComputePkg        *compute_pkg;
                          
   hypre_CommHandle        *comm_handle;
                          
   hypre_BoxArrayArray     *compute_box_aa;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;
                          
   hypre_Box               *A_data_box;
   hypre_Box               *x_data_box;
   hypre_Box               *y_data_box;
                          
   int                      Ai;
   int                      xi;
   int                      xoff0;
   int                      xoff1;
   int                      xoff2;
   int                      xoff3;
   int                      xoff4;
   int                      xoff5;
   int                      xoff6;
   int                      yi;
                          
   double                  *Ap0;
   double                  *Ap1;
   double                  *Ap2;
   double                  *Ap3;
   double                  *Ap4;
   double                  *Ap5;
   double                  *Ap6;
   double                  *xp;
   double                  *yp;
                          
   hypre_BoxArray          *boxes;
   hypre_Box               *box;
   hypre_Index              loop_size;
   hypre_IndexRef           start;
   hypre_IndexRef           stride;
                          
   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   int                      stencil_size;
   int                      depth;
                          
   double                   temp;
   int                      compute_i, i, j, si;
   int                      loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   compute_pkg = (matvec_data -> compute_pkg);

   stride = hypre_ComputePkgStride(compute_pkg);

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

            hypre_BoxGetSize(box, loop_size);

            hypre_BoxLoop1Begin(loop_size,
                                y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, yi)
               {
                  yp[yi] *= beta;
               }
            hypre_BoxLoop1End(yi);
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
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

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
                        hypre_BoxGetSize(box, loop_size);

                        hypre_BoxLoop1Begin(loop_size,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop1For(loopi, loopj, loopk, yi)
                           {
                              yp[yi] = 0.0;
                           }
                        hypre_BoxLoop1End(yi);
                     }
                     else
                     {
                        hypre_BoxGetSize(box, loop_size);

                        hypre_BoxLoop1Begin(loop_size,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop1For(loopi, loopj, loopk, yi)
                           {
                              yp[yi] *= temp;
                           }
                        hypre_BoxLoop1End(yi);
                     }
                  }
            }
         }
         break;

         case 1:
         {
            hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * y += A*x
       *--------------------------------------------------------------------*/

      hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

            xp = hypre_StructVectorBoxData(x, i);
            yp = hypre_StructVectorBoxData(y, i);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  hypre_BoxGetSize(compute_box, loop_size);
                  start  = hypre_BoxIMin(compute_box);

                  /* unroll up to depth MAX_DEPTH */
                  for (si = 0; si < stencil_size; si+= MAX_DEPTH)
                  {
                     depth = hypre_min(MAX_DEPTH, (stencil_size -si));
                     switch(depth)
                     {
                        case 7:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);
                        Ap2 = hypre_StructMatrixBoxData(A, i, si+2);
                        Ap3 = hypre_StructMatrixBoxData(A, i, si+3);
                        Ap4 = hypre_StructMatrixBoxData(A, i, si+4);
                        Ap5 = hypre_StructMatrixBoxData(A, i, si+5);
                        Ap6 = hypre_StructMatrixBoxData(A, i, si+6);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);
                        xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+2]);
                        xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+3]);
                        xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+4]);
                        xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+5]);
                        xoff6 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+6]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1] +
                                 Ap2[Ai] * xp[xi + xoff2] +
                                 Ap3[Ai] * xp[xi + xoff3] +
                                 Ap4[Ai] * xp[xi + xoff4] +
                                 Ap5[Ai] * xp[xi + xoff5] +
                                 Ap6[Ai] * xp[xi + xoff6];
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;

                        case 6:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);
                        Ap2 = hypre_StructMatrixBoxData(A, i, si+2);
                        Ap3 = hypre_StructMatrixBoxData(A, i, si+3);
                        Ap4 = hypre_StructMatrixBoxData(A, i, si+4);
                        Ap5 = hypre_StructMatrixBoxData(A, i, si+5);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);
                        xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+2]);
                        xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+3]);
                        xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+4]);
                        xoff5 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+5]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1] +
                                 Ap2[Ai] * xp[xi + xoff2] +
                                 Ap3[Ai] * xp[xi + xoff3] +
                                 Ap4[Ai] * xp[xi + xoff4] +
                                 Ap5[Ai] * xp[xi + xoff5];
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);
                        
                        break;

                        case 5:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);
                        Ap2 = hypre_StructMatrixBoxData(A, i, si+2);
                        Ap3 = hypre_StructMatrixBoxData(A, i, si+3);
                        Ap4 = hypre_StructMatrixBoxData(A, i, si+4);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);
                        xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+2]);
                        xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+3]);
                        xoff4 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+4]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1] +
                                 Ap2[Ai] * xp[xi + xoff2] +
                                 Ap3[Ai] * xp[xi + xoff3] +
                                 Ap4[Ai] * xp[xi + xoff4]; 
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;

                        case 4:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);
                        Ap2 = hypre_StructMatrixBoxData(A, i, si+2);
                        Ap3 = hypre_StructMatrixBoxData(A, i, si+3);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);
                        xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+2]);
                        xoff3 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+3]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1] +
                                 Ap2[Ai] * xp[xi + xoff2] +
                                 Ap3[Ai] * xp[xi + xoff3];
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;

                        case 3:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);
                        Ap2 = hypre_StructMatrixBoxData(A, i, si+2);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);
                        xoff2 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+2]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1] +
                                 Ap2[Ai] * xp[xi + xoff2]; 
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;

                        case 2:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);
                        Ap1 = hypre_StructMatrixBoxData(A, i, si+1);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);
                        xoff1 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+1]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0] +
                                 Ap1[Ai] * xp[xi + xoff1];
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;

                        case 1:
                        Ap0 = hypre_StructMatrixBoxData(A, i, si+0);

                        xoff0 = hypre_BoxOffsetDistance(x_data_box,
                                                        stencil_shape[si+0]);

                        hypre_BoxLoop3Begin(loop_size,
                                            A_data_box, start, stride, Ai,
                                            x_data_box, start, stride, xi,
                                            y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
                           {
                              yp[yi] +=
                                 Ap0[Ai] * xp[xi + xoff0];
                           }
                        hypre_BoxLoop3End(Ai, xi, yi);

                        break;
                     }
                  }

                  if (alpha != 1.0)
                  {
                     hypre_BoxLoop1Begin(loop_size,
                                         y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop1For(loopi, loopj, loopk, yi)
                        {
                           yp[yi] *= alpha;
                        }
                     hypre_BoxLoop1End(yi);
                  }
               }
         }
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
   int ierr = 0;

   hypre_StructMatvecData *matvec_data = matvec_vdata;

   if (matvec_data)
   {
      hypre_StructMatrixDestroy(matvec_data -> A);
      hypre_StructVectorDestroy(matvec_data -> x);
      hypre_ComputePkgDestroy(matvec_data -> compute_pkg );
      hypre_TFree(matvec_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvec
 *--------------------------------------------------------------------------*/

int
hypre_StructMatvec( double              alpha,
                    hypre_StructMatrix *A,
                    hypre_StructVector *x,
                    double              beta,
                    hypre_StructVector *y     )
{
   int ierr = 0;

   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   ierr = hypre_StructMatvecSetup(matvec_data, A, x);
   ierr = hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   ierr = hypre_StructMatvecDestroy(matvec_data);

   return ierr;
}

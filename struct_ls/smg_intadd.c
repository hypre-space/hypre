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

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif
#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGIntAddData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix *PT;
   hypre_ComputePkg   *compute_pkg;
   hypre_BoxArray     *coarse_points;
   hypre_Index         cindex;
   hypre_Index         stride;

   int               time_index;

} hypre_SMGIntAddData;

/*--------------------------------------------------------------------------
 * hypre_SMGIntAddInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_SMGIntAddInitialize( )
{
   hypre_SMGIntAddData *intadd_data;

   intadd_data = hypre_CTAlloc(hypre_SMGIntAddData, 1);
   (intadd_data -> time_index)  = hypre_InitializeTiming("SMGIntAdd");

   return (void *) intadd_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGIntAddSetup
 *--------------------------------------------------------------------------*/

int
hypre_SMGIntAddSetup( void               *intadd_vdata,
                      hypre_StructMatrix *PT,
                      hypre_StructVector *xc,
                      hypre_StructVector *e,
                      hypre_StructVector *x,
                      hypre_Index         cindex,
                      hypre_Index         findex,
                      hypre_Index         stride      )
{
   hypre_SMGIntAddData    *intadd_data = intadd_vdata;

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil_PT;
   hypre_Index            *stencil_PT_shape;
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;
   int                     stencil_size;
   int                     stencil_dim;
                       
   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **temp_send_processes;
   int                   **temp_recv_processes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
   hypre_BoxArrayArray    *f_send_boxes;
   hypre_BoxArrayArray    *f_recv_boxes;
                       
   hypre_ComputePkg       *compute_pkg;
   hypre_BoxArray         *coarse_points;

   int                     i;
   int                     ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid = hypre_StructVectorGrid(x);

   /*----------------------------------------------------------
    * Use PT-stencil-element-1 to set up the compute package
    *----------------------------------------------------------*/

   stencil_PT = hypre_StructMatrixStencil(PT);
   stencil_PT_shape = hypre_StructStencilShape(stencil_PT);
   stencil_size = 1;
   stencil_dim = hypre_StructStencilDim(stencil_PT);
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   hypre_CopyIndex(stencil_PT_shape[1], stencil_shape[0]);
   stencil = hypre_NewStructStencil(stencil_dim, stencil_size, stencil_shape);

   hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                        &temp_send_processes, &temp_recv_processes,
                        &indt_boxes, &dept_boxes,
                        grid, stencil);

   hypre_FreeStructStencil(stencil);

   /*----------------------------------------------------------
    * Project sends and recieves to fine and coarse points
    *----------------------------------------------------------*/

   f_send_boxes = hypre_DuplicateBoxArrayArray(send_boxes);
   f_recv_boxes = hypre_DuplicateBoxArrayArray(recv_boxes);
   hypre_ProjectBoxArrayArray(f_send_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(f_recv_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(send_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(recv_boxes, cindex, stride);
   hypre_ProjectBoxArrayArray(indt_boxes, findex, stride);
   hypre_ProjectBoxArrayArray(dept_boxes, findex, stride);

   /*----------------------------------------------------------
    * Reverse sends and recieves for fine points, append to
    * sends and recieves for coarse points. 
    *----------------------------------------------------------*/

   hypre_AppendBoxArrayArrayAndProcs(temp_recv_processes, temp_send_processes,
                                     f_recv_boxes, send_boxes,
                                     &send_processes);
   hypre_AppendBoxArrayArrayAndProcs(temp_send_processes, temp_recv_processes,
                                     f_send_boxes, recv_boxes,
                                     &recv_processes);

   hypre_ForBoxArrayI(i, f_send_boxes)
      {
         hypre_FreeBoxArrayShell(hypre_BoxArrayArrayBoxArray(f_send_boxes, i));
         hypre_TFree(temp_send_processes[i]);
      }
   hypre_FreeBoxArrayArrayShell(f_send_boxes);
   hypre_ForBoxArrayI(i, f_recv_boxes)
      {
         hypre_FreeBoxArrayShell(hypre_BoxArrayArrayBoxArray(f_recv_boxes, i));
         hypre_TFree(temp_recv_processes[i]);
      }
   hypre_FreeBoxArrayArrayShell(f_recv_boxes);

   hypre_TFree(temp_send_processes);
   hypre_TFree(temp_recv_processes);

   compute_pkg = hypre_NewComputePkg(send_boxes, recv_boxes,
                                     stride, stride,
                                     send_processes, recv_processes,
                                     indt_boxes, dept_boxes,
                                     stride, grid,
                                     hypre_StructVectorDataSpace(e), 1);

   /*----------------------------------------------------------
    * Set up the coarse points BoxArray
    *----------------------------------------------------------*/

   coarse_points = hypre_DuplicateBoxArray(hypre_StructGridBoxes(grid));
   hypre_ProjectBoxArray(coarse_points, cindex, stride);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (intadd_data -> PT)            = PT;
   (intadd_data -> compute_pkg)   = compute_pkg;
   (intadd_data -> coarse_points) = coarse_points;
   hypre_CopyIndex(cindex, (intadd_data -> cindex));
   hypre_CopyIndex(stride, (intadd_data -> stride));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGIntAdd:
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
hypre_SMGIntAdd( void               *intadd_vdata,
                 hypre_StructMatrix *PT,
                 hypre_StructVector *xc,
                 hypre_StructVector *e,
                 hypre_StructVector *x           )
{
   int ierr = 0;

   hypre_SMGIntAddData    *intadd_data = intadd_vdata;

   hypre_ComputePkg       *compute_pkg;
   hypre_BoxArray         *coarse_points;
   hypre_IndexRef          cindex;
   hypre_IndexRef          stride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *PT_data_box;
   hypre_Box              *xc_data_box;
   hypre_Box              *x_data_box;
   hypre_Box              *e_data_box;
                       
   int                     PTi;
   int                     xci;
   int                     xi;
   int                     ei;
                         
   double                 *xcp;
   double                 *PTp0, *PTp1;
   double                 *ep, *ep0, *ep1;
   double                 *xp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
   hypre_Index             startc;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   int                     compute_i, i, j;
   int                     loopi, loopj, loopk;

   hypre_BeginTiming(intadd_data -> time_index);

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   compute_pkg   = (intadd_data -> compute_pkg);
   coarse_points = (intadd_data -> coarse_points);
   cindex        = (intadd_data -> cindex);
   stride        = (intadd_data -> stride);

   stencil       = hypre_StructMatrixStencil(PT);
   stencil_shape = hypre_StructStencilShape(stencil);

   hypre_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Compute e = (P^T)_off x_c, where (P^T)_off corresponds to the
    * off-diagonal coefficients of P^T.  Interleave the results.
    *--------------------------------------------------------------------*/

   hypre_ClearStructVectorAllValues(e);

   compute_box_a = coarse_points;
   hypre_ForBoxI(i, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, i);

         start = hypre_BoxIMin(compute_box);
         hypre_SMGMapFineToCoarse(start, startc, cindex, stride);

         e_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
         PT_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(PT), i);
         xc_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), i);

         ep0 = hypre_StructVectorBoxData(e, i);
         ep1 = hypre_StructVectorBoxData(e, i) +
            hypre_BoxOffsetDistance(e_data_box, stencil_shape[1]);
         PTp0 = hypre_StructMatrixBoxData(PT, i, 0);
         PTp1 = hypre_StructMatrixBoxData(PT, i, 1);
         xcp = hypre_StructVectorBoxData(xc, i);

         hypre_GetStrideBoxSize(compute_box, stride, loop_size);
         hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
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
            ep = hypre_StructVectorData(e);
            comm_handle = hypre_InitializeIndtComputations(compute_pkg, ep);
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

      /*----------------------------------------
       * Compute x at coarse points.
       *----------------------------------------*/

      if (compute_i == 0)
      {
         compute_box_a = coarse_points;
         hypre_ForBoxI(i, compute_box_a)
            {
               compute_box = hypre_BoxArrayBox(compute_box_a, i);

               start = hypre_BoxIMin(compute_box);
               hypre_SMGMapFineToCoarse(start, startc, cindex, stride);

               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
               xc_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), i);

               xp = hypre_StructVectorBoxData(x, i);
               xcp = hypre_StructVectorBoxData(xc, i);

               hypre_GetStrideBoxSize(compute_box, stride, loop_size);
               hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
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

      hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            x_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            e_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);

            xp  = hypre_StructVectorBoxData(x, i);
            ep0 = hypre_StructVectorBoxData(e, i);
            ep1 = hypre_StructVectorBoxData(e, i) +
               hypre_BoxOffsetDistance(e_data_box, stencil_shape[1]);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);

                  hypre_GetStrideBoxSize(compute_box, stride, loop_size);
                  hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
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

   hypre_IncFLOPCount(5*hypre_StructVectorGlobalSize(xc));
   hypre_EndTiming(intadd_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGIntAddFinalize
 *--------------------------------------------------------------------------*/

int
hypre_SMGIntAddFinalize( void *intadd_vdata )
{
   int ierr = 0;

   hypre_SMGIntAddData *intadd_data = intadd_vdata;

   if (intadd_data)
   {
      hypre_FreeBoxArray(intadd_data -> coarse_points);
      hypre_FreeComputePkg(intadd_data -> compute_pkg);
      hypre_FinalizeTiming(intadd_data -> time_index);
      hypre_TFree(intadd_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AppendBoxArrayArrayAndProcs:
 *   Append box_array_array_0 to box_array_array_1.
 *   The two BoxArrayArrays must be the same length.
 *   Additionally create an appended version of processes.
 *--------------------------------------------------------------------------*/
 
int
hypre_AppendBoxArrayArrayAndProcs( int                  **processes_0,
                                   int                  **processes_1,
                                   hypre_BoxArrayArray   *box_array_array_0,
                                   hypre_BoxArrayArray   *box_array_array_1,
                                   int                 ***processes_ptr     )
{
   int                ierr = 0;

   int              **processes;
   int                box_array_array_size; 
   hypre_BoxArray    *box_array_0;
   hypre_BoxArray    *box_array_1;
   int                box_array_size_0; 
   int                box_array_size_1; 
   int                i;
   int                j;
   int                k;
 
   box_array_array_size = hypre_BoxArrayArraySize(box_array_array_0);
   processes = hypre_CTAlloc(int *, box_array_array_size);

   hypre_ForBoxArrayI(i, box_array_array_0)
      {
         box_array_0 = hypre_BoxArrayArrayBoxArray(box_array_array_0, i);  
         box_array_1 = hypre_BoxArrayArrayBoxArray(box_array_array_1, i);  
         box_array_size_0 = hypre_BoxArraySize(box_array_0);
         box_array_size_1 = hypre_BoxArraySize(box_array_1);
         processes[i] =
            hypre_CTAlloc(int, box_array_size_0 + box_array_size_1);
         for ( j=0 ; j < box_array_size_1; j++)
            processes[i][j] = processes_1[i][j];
         for ( k=0 ; k < box_array_size_0; k++)
            processes[i][k+box_array_size_1] = processes_0[i][k];
         hypre_AppendBoxArray(box_array_0, box_array_1);
      }

   *processes_ptr = processes;

   return ierr;
}


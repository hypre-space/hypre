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
   hypre_SBoxArray    *coarse_points;
   hypre_Index         cindex;
   hypre_Index         cstride;

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
#ifdef HYPRE_USE_PTHREADS
   hypre_TimingThreadWrapper({
      (intadd_data -> time_index)  = hypre_InitializeTiming("SMGIntAdd");
   });
#else
   (intadd_data -> time_index)  = hypre_InitializeTiming("SMGIntAdd");
#endif
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
                      hypre_Index         cstride,
                      hypre_Index         findex,
                      hypre_Index         fstride      )
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
                       
   hypre_SBoxArrayArray   *f_send_sboxes;
   hypre_SBoxArrayArray   *f_recv_sboxes;
   hypre_SBoxArrayArray   *send_sboxes;
   hypre_SBoxArrayArray   *recv_sboxes;
   hypre_SBoxArrayArray   *indt_sboxes;
   hypre_SBoxArrayArray   *dept_sboxes;
                       
   hypre_ComputePkg       *compute_pkg;
   hypre_SBoxArray        *coarse_points;

   int                     i;
   int                     ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructVectorGrid(x);

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

   /*----------------------------------------------------------
    * Project sends and recieves to fine and coarse points
    *----------------------------------------------------------*/

   f_send_sboxes = hypre_ProjectBoxArrayArray(send_boxes, findex, fstride);
   f_recv_sboxes = hypre_ProjectBoxArrayArray(recv_boxes, findex, fstride);
   send_sboxes = hypre_ProjectBoxArrayArray(send_boxes, cindex, cstride);
   recv_sboxes = hypre_ProjectBoxArrayArray(recv_boxes, cindex, cstride);
   indt_sboxes = hypre_ProjectBoxArrayArray(indt_boxes, findex, fstride);
   dept_sboxes = hypre_ProjectBoxArrayArray(dept_boxes, findex, fstride);

   hypre_FreeBoxArrayArray(send_boxes);
   hypre_FreeBoxArrayArray(recv_boxes);
   hypre_FreeBoxArrayArray(indt_boxes);
   hypre_FreeBoxArrayArray(dept_boxes);
   hypre_FreeStructStencil(stencil);

   /*----------------------------------------------------------
    * Reverse sends and recieves for fine points, append to
    * sends and recieves for coarse points. 
    *----------------------------------------------------------*/

   hypre_AppendSBoxArrayArrayAndProcs(temp_recv_processes, temp_send_processes,
                                      f_recv_sboxes, send_sboxes,
                                      &send_processes);
   hypre_AppendSBoxArrayArrayAndProcs(temp_send_processes, temp_recv_processes,
                                      f_send_sboxes, recv_sboxes,
                                      &recv_processes);

   hypre_ForSBoxArrayI(i, f_send_sboxes)
      {
         hypre_FreeSBoxArrayShell(hypre_SBoxArrayArraySBoxArray(
            f_send_sboxes, i));
         hypre_TFree(temp_send_processes[i]);
      }
   hypre_FreeSBoxArrayArrayShell(f_send_sboxes);
   hypre_ForSBoxArrayI(i, f_recv_sboxes)
      {
         hypre_FreeSBoxArrayShell(hypre_SBoxArrayArraySBoxArray(
            f_recv_sboxes, i));
         hypre_TFree(temp_recv_processes[i]);
      }
   hypre_FreeSBoxArrayArrayShell(f_recv_sboxes);

   hypre_TFree(temp_send_processes);
   hypre_TFree(temp_recv_processes);

   compute_pkg = hypre_NewComputePkg(send_sboxes, recv_sboxes,
                                     send_processes, recv_processes,
                                     indt_sboxes, dept_sboxes,
                                     grid, hypre_StructVectorDataSpace(e), 1);

   /*----------------------------------------------------------
    * Set up the coarse points SBoxArray
    *----------------------------------------------------------*/

   coarse_points = hypre_ProjectBoxArray(hypre_StructGridBoxes(grid),
                                         cindex, cstride);

   /*----------------------------------------------------------
    * Set up the intadd data structure
    *----------------------------------------------------------*/

   (intadd_data -> PT)            = PT;
   (intadd_data -> compute_pkg)   = compute_pkg;
   (intadd_data -> coarse_points) = coarse_points;
   hypre_CopyIndex(cindex, (intadd_data -> cindex));
   hypre_CopyIndex(cstride, (intadd_data -> cstride));

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
   hypre_SBoxArray        *coarse_points;
   hypre_IndexRef          cindex;
   hypre_IndexRef          cstride;

   hypre_CommHandle       *comm_handle;
                       
   hypre_SBoxArrayArray   *compute_sbox_aa;
   hypre_SBoxArray        *compute_sbox_a;
   hypre_SBox             *compute_sbox;
                       
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
   hypre_IndexRef          stride;
   hypre_Index             startc;
   hypre_Index             stridec;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;

   int                     compute_i, i, j;
   int                     loopi, loopj, loopk;

#ifdef HYPRE_USE_PTHREADS
   hypre_TimingThreadWrapper({
      hypre_BeginTiming(intadd_data -> time_index);
   });
#else
   hypre_BeginTiming(intadd_data -> time_index);
#endif

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   compute_pkg   = (intadd_data -> compute_pkg);
   coarse_points = (intadd_data -> coarse_points);
   cindex        = (intadd_data -> cindex);
   cstride       = (intadd_data -> cstride);

   stencil       = hypre_StructMatrixStencil(PT);
   stencil_shape = hypre_StructStencilShape(stencil);

   stride = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Compute e = (P^T)_off x_c, where (P^T)_off corresponds to the
    * off-diagonal coefficients of P^T.  Interleave the results.
    *--------------------------------------------------------------------*/

   hypre_ClearStructVectorAllValues(e);

   compute_sbox_a = coarse_points;
   hypre_ForSBoxI(i, compute_sbox_a)
      {
         compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);

         start = hypre_SBoxIMin(compute_sbox);
         hypre_SMGMapFineToCoarse(start, startc, cindex, cstride);

         e_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
         PT_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(PT), i);
         xc_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), i);

         ep0 = hypre_StructVectorBoxData(e, i);
         ep1 = hypre_StructVectorBoxData(e, i) +
            hypre_BoxOffsetDistance(e_data_box, stencil_shape[1]);
         PTp0 = hypre_StructMatrixBoxData(PT, i, 0);
         PTp1 = hypre_StructMatrixBoxData(PT, i, 1);
         xcp = hypre_StructVectorBoxData(xc, i);

         hypre_GetSBoxSize(compute_sbox, loop_size);
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

      /*----------------------------------------
       * Compute x at coarse points.
       *----------------------------------------*/

      if (compute_i == 0)
      {
         compute_sbox_a = coarse_points;
         hypre_ForSBoxI(i, compute_sbox_a)
            {
               compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);

               start = hypre_SBoxIMin(compute_sbox);
               hypre_SMGMapFineToCoarse(start, startc, cindex, cstride);

               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
               xc_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc), i);

               xp = hypre_StructVectorBoxData(x, i);
               xcp = hypre_StructVectorBoxData(xc, i);

               hypre_GetSBoxSize(compute_sbox, loop_size);
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

      hypre_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

            x_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            e_data_box  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);

            xp  = hypre_StructVectorBoxData(x, i);
            ep0 = hypre_StructVectorBoxData(e, i);
            ep1 = hypre_StructVectorBoxData(e, i) +
               hypre_BoxOffsetDistance(e_data_box, stencil_shape[1]);

            hypre_ForSBoxI(j, compute_sbox_a)
               {
                  compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

                  start  = hypre_SBoxIMin(compute_sbox);

                  hypre_GetSBoxSize(compute_sbox, loop_size);
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

#ifdef HYPRE_USE_PTHREADS
   hypre_TimingThreadWrapper({
      hypre_IncFLOPCount(5*hypre_StructVectorGlobalSize(xc));
      hypre_EndTiming(intadd_data -> time_index);
   });
#else
   hypre_IncFLOPCount(5*hypre_StructVectorGlobalSize(xc));
   hypre_EndTiming(intadd_data -> time_index);
#endif


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
      hypre_FreeSBoxArray(intadd_data -> coarse_points);
      hypre_FreeComputePkg(intadd_data -> compute_pkg);
#ifdef HYPRE_USE_PTHREADS
      hypre_TimingThreadWrapper({
         hypre_FinalizeTiming(intadd_data -> time_index);
      });
#else
      hypre_FinalizeTiming(intadd_data -> time_index);
#endif
      hypre_TFree(intadd_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AppendSBoxArrayArrayAndProcs:
 *   Append sbox_array_array_0 to sbox_array_array_1.
 *   The two SBoxArrayArrays must be the same length.
 *   Additionally create an appended version of processes.
 *--------------------------------------------------------------------------*/
 
void
hypre_AppendSBoxArrayArrayAndProcs( int                   **processes_0,
                                    int                   **processes_1,
                                    hypre_SBoxArrayArray   *sbox_array_array_0,
                                    hypre_SBoxArrayArray   *sbox_array_array_1,
                                    int                  ***processes_ptr)
{
   int              **processes;
   int                sbox_array_array_size; 
   hypre_SBoxArray   *sbox_array_0;
   hypre_SBoxArray   *sbox_array_1;
   int                sbox_array_size_0; 
   int                sbox_array_size_1; 
   int                i;
   int                j;
   int                k;
 
   sbox_array_array_size = hypre_SBoxArrayArraySize(sbox_array_array_0);
   processes = hypre_CTAlloc(int *, sbox_array_array_size);

   hypre_ForSBoxArrayI(i, sbox_array_array_0)
      {
         sbox_array_0 = hypre_SBoxArrayArraySBoxArray(sbox_array_array_0, i);  
         sbox_array_1 = hypre_SBoxArrayArraySBoxArray(sbox_array_array_1, i);  
         sbox_array_size_0 = hypre_SBoxArraySize(sbox_array_0);
         sbox_array_size_1 = hypre_SBoxArraySize(sbox_array_1);
         processes[i] =
            hypre_CTAlloc(int, sbox_array_size_0 + sbox_array_size_1);
         for ( j=0 ; j < sbox_array_size_1; j++)
            processes[i][j] = processes_1[i][j];
         for ( k=0 ; k < sbox_array_size_0; k++)
            processes[i][k+sbox_array_size_1] = processes_0[i][k];
         hypre_AppendSBoxArray(sbox_array_0, sbox_array_1);
      }

   *processes_ptr = processes;
}


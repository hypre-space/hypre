/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 * Cyclic reduction algorithm (coded as if it were a 1D MG method)
 *
 *****************************************************************************/

#include "headers.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define hypre_CycRedSetCIndex(base_index, base_stride, level, cdir, cindex) \
{\
   if (level > 0)\
      hypre_SetIndex(cindex, 0, 0, 0);\
   else\
      hypre_CopyIndex(base_index,  cindex);\
   hypre_IndexD(cindex, cdir) += 0;\
}

#define hypre_CycRedSetFIndex(base_index, base_stride, level, cdir, findex) \
{\
   if (level > 0)\
      hypre_SetIndex(findex, 0, 0, 0);\
   else\
      hypre_CopyIndex(base_index,  findex);\
   hypre_IndexD(findex, cdir) += 1;\
}

#define hypre_CycRedSetStride(base_index, base_stride, level, cdir, stride) \
{\
   if (level > 0)\
      hypre_SetIndex(stride, 1, 1, 1);\
   else\
      hypre_CopyIndex(base_stride, stride);\
   hypre_IndexD(stride, cdir) *= 2;\
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
                      
   int                   num_levels;
                      
   int                   cdir;         /* coarsening direction */
   hypre_Index           base_index;
   hypre_Index           base_stride;

   hypre_StructGrid    **grid_l;
                    
   hypre_BoxArray       *base_points;
   hypre_BoxArray      **fine_points_l;

   double               *data;
   hypre_StructMatrix  **A_l;
   hypre_StructVector  **x_l;

   hypre_ComputePkg    **down_compute_pkg_l;
   hypre_ComputePkg    **up_compute_pkg_l;

   int                   time_index;
   int                   solve_flops;

} hypre_CyclicReductionData;

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CyclicReductionCreate( MPI_Comm  comm )
{
   hypre_CyclicReductionData *cyc_red_data;

   cyc_red_data = hypre_CTAlloc(hypre_CyclicReductionData, 1);
   
   (cyc_red_data -> comm) = comm;
   (cyc_red_data -> cdir) = 0;
   (cyc_red_data -> time_index)  = hypre_InitializeTiming("CyclicReduction");

   /* set defaults */
   hypre_SetIndex((cyc_red_data -> base_index), 0, 0, 0);
   hypre_SetIndex((cyc_red_data -> base_stride), 1, 1, 1);

   return (void *) cyc_red_data;
}

/*--------------------------------------------------------------------------
 * hypre_CycRedCreateCoarseOp
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_CycRedCreateCoarseOp( hypre_StructMatrix *A,
                            hypre_StructGrid   *coarse_grid,
                            int                 cdir        )
{
   hypre_StructMatrix    *Ac;

   hypre_Index           *Ac_stencil_shape;
   hypre_StructStencil   *Ac_stencil;
   int                    Ac_stencil_size;
   int                    Ac_stencil_dim;
   int                    Ac_num_ghost[] = {0, 0, 0, 0, 0, 0};
                       
   int                    i;
   int                    stencil_rank;
 
   Ac_stencil_dim = 1;

   /*-----------------------------------------------
    * Define Ac_stencil
    *-----------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------
    * non-symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *-----------------------------------------------*/

   if (!hypre_StructMatrixSymmetric(A))
   {
      Ac_stencil_size = 3;
      Ac_stencil_shape = hypre_CTAlloc(hypre_Index, Ac_stencil_size);
      for (i = -1; i < 2; i++)
      {
         /* Storage for 3 elements (c,w,e) */
         hypre_SetIndex(Ac_stencil_shape[stencil_rank],i,0,0);
         stencil_rank++;
      }
   }

   /*-----------------------------------------------
    * symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *
    * Only store the lower triangular part + diagonal = 2 entries,
    * lower triangular means the lower triangular part on the matrix
    * in the standard lexicalgraphic ordering.
    *-----------------------------------------------*/

   else
   {
      Ac_stencil_size = 2;
      Ac_stencil_shape = hypre_CTAlloc(hypre_Index, Ac_stencil_size);
      for (i = -1; i < 1; i++)
      {

         /* Storage for 2 elements in (c,w) */
         hypre_SetIndex(Ac_stencil_shape[stencil_rank],i,0,0);
         stencil_rank++;
      }
   }

   Ac_stencil = hypre_StructStencilCreate(Ac_stencil_dim, Ac_stencil_size,
                                       Ac_stencil_shape);

   Ac = hypre_StructMatrixCreate(hypre_StructMatrixComm(A),
                              coarse_grid, Ac_stencil);

   hypre_StructStencilDestroy(Ac_stencil);

   /*-----------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------*/

   hypre_StructMatrixSymmetric(Ac) = hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------
    * Set number of ghost points
    *-----------------------------------------------*/

   Ac_num_ghost[2*cdir]     = 1;
   if (!hypre_StructMatrixSymmetric(A))
   {
     Ac_num_ghost[2*cdir + 1] = 1;
   }
   hypre_StructMatrixSetNumGhost(Ac, Ac_num_ghost);

   hypre_StructMatrixInitializeShell(Ac);
 
   return Ac;
}

/*--------------------------------------------------------------------------
 * hypre_CycRedSetupCoarseOp
 *--------------------------------------------------------------------------*/

int
hypre_CycRedSetupCoarseOp( hypre_StructMatrix *A,
                           hypre_StructMatrix *Ac,
                           hypre_Index         cindex,
                           hypre_Index         cstride )

{
   hypre_Index             index;

   hypre_StructGrid       *fgrid;
   int                    *fgrid_ids;
   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   int                    *cgrid_ids;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             fstart;
   hypre_IndexRef          stridef;
   hypre_Index             loop_size;

   int                     fi, ci;
   int                     loopi, loopj, loopk;

   hypre_Box              *A_dbox;
   hypre_Box              *Ac_dbox;

   double                 *a_cc, *a_cw, *a_ce;
   double                 *ac_cc, *ac_cw, *ac_ce;
                    
   int                     iA, iAm1, iAp1;
   int                     iAc;
                         
   int                     xOffsetA; 
                         
   int                     ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(Ac);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = hypre_BoxIMin(cgrid_box);
         hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
         Ac_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Ac), ci);

         /*-----------------------------------------------
          * Extract pointers for 3-point fine grid operator:
          * 
          * a_cc is pointer for center coefficient
          * a_cw is pointer for west coefficient
          * a_ce is pointer for east coefficient
          *-----------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,-1,0,0);
         a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,1,0,0);
         a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         /*-----------------------------------------------
          * Extract pointers for coarse grid operator - always 3-point:
          *
          * If A is symmetric so is Ac.  We build only the
          * lower triangular part (plus diagonal).
          * 
          * ac_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         ac_cc = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         hypre_SetIndex(index,-1,0,0);
         ac_cw = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_SetIndex(index,1,0,0);
            ac_ce = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
         }

         /*-----------------------------------------------
          * Define offsets for fine grid stencil and interpolation
          *
          * In the BoxLoop below I assume iA and iP refer
          * to data associated with the point which we are
          * building the stencil for.  The below offsets
          * are used in refering to data associated with
          * other points. 
          *-----------------------------------------------*/

         hypre_SetIndex(index,1,0,0);
         xOffsetA = hypre_BoxOffsetDistance(A_dbox,index); 

         /*-----------------------------------------------
          * non-symmetric case
          *-----------------------------------------------*/

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                A_dbox, fstart, stridef, iA,
                                Ac_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA,iAc,iAm1,iAp1
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, iA, iAc)
               {
                  iAm1 = iA - xOffsetA;
                  iAp1 = iA + xOffsetA;

                  ac_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                  ac_cc[iAc] = a_cc[iA]
                     - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                     - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   

                  ac_ce[iAc] = - a_ce[iA] *a_ce[iAp1] / a_cc[iAp1];

               }
            hypre_BoxLoop2End(iA, iAc);
         }

         /*-----------------------------------------------
          * symmetric case
          *-----------------------------------------------*/

         else
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                A_dbox, fstart, stridef, iA,
                                Ac_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA,iAc,iAm1,iAp1
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, iA, iAc)
               {
                  iAm1 = iA - xOffsetA;
                  iAp1 = iA + xOffsetA;

                  ac_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                  ac_cc[iAc] = a_cc[iA]
                     - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                     - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   
               }
            hypre_BoxLoop2End(iA, iAc);
         }

      } /* end ForBoxI */

   hypre_StructMatrixAssemble(Ac);

   /*-----------------------------------------------------------------------
    * Collapse stencil in periodic direction on coarsest grid.
    *-----------------------------------------------------------------------*/

   if (hypre_IndexX(hypre_StructGridPeriodic(cgrid)) == 1)
   {
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = hypre_BoxIMin(cgrid_box);

         Ac_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Ac), ci);

         /*-----------------------------------------------
          * Extract pointers for coarse grid operator - always 3-point:
          *
          * If A is symmetric so is Ac.  We build only the
          * lower triangular part (plus diagonal).
          *
          * ac_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         ac_cc = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         hypre_SetIndex(index,-1,0,0);
         ac_cw = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_SetIndex(index,1,0,0);
            ac_ce = hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
         }


         /*-----------------------------------------------
          * non-symmetric case
          *-----------------------------------------------*/

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

            hypre_BoxLoop1Begin(loop_size,
                                Ac_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
               {
                  ac_cc[iAc] += (ac_cw[iAc] + ac_ce[iAc]);
                  ac_cw[iAc]  =  0.0;
                  ac_ce[iAc]  =  0.0;
               }
            hypre_BoxLoop1End(iAc);
         }

         /*-----------------------------------------------
          * symmetric case
          *-----------------------------------------------*/

         else
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

            hypre_BoxLoop1Begin(loop_size,
                                Ac_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
               {
                  ac_cc[iAc] += (2.0  *  ac_cw[iAc]);
                  ac_cw[iAc]  =  0.0;
               }
            hypre_BoxLoop1End(iAc);
         }

      } /* end ForBoxI */

   }

   hypre_StructMatrixAssemble(Ac);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionSetup
 *--------------------------------------------------------------------------*/

int
hypre_CyclicReductionSetup( void               *cyc_red_vdata,
                            hypre_StructMatrix *A,
                            hypre_StructVector *b,
                            hypre_StructVector *x             )
{
   hypre_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   MPI_Comm                comm        = (cyc_red_data -> comm);
   int                     cdir        = (cyc_red_data -> cdir);
   hypre_IndexRef          base_index  = (cyc_red_data -> base_index);
   hypre_IndexRef          base_stride = (cyc_red_data -> base_stride);

   int                     num_levels;
   hypre_StructGrid      **grid_l;
   hypre_BoxArray         *base_points;
   hypre_BoxArray        **fine_points_l;
   double                 *data;
   int                     data_size = 0;
   hypre_StructMatrix    **A_l;
   hypre_StructVector    **x_l;
   hypre_ComputePkg      **down_compute_pkg_l;
   hypre_ComputePkg      **up_compute_pkg_l;

   hypre_Index             cindex;
   hypre_Index             findex;
   hypre_Index             stride;

   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_StructGrid       *grid;

   hypre_Box              *cbox;
                    
   int                     l;
   int                     flop_divisor;
                         
   int                     x_num_ghost[] = {0, 0, 0, 0, 0, 0};
                         
   int                     ierr = 0;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid = hypre_StructMatrixGrid(A);

   /* Compute a preliminary num_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
   num_levels = hypre_Log2(hypre_BoxSizeD(cbox, cdir)) + 2;

   grid_l    = hypre_TAlloc(hypre_StructGrid *, num_levels);
   hypre_StructGridRef(grid, &grid_l[0]);
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      if ( hypre_BoxIMinD(cbox, cdir) == hypre_BoxIMaxD(cbox, cdir) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen cbox */
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                  hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                  hypre_BoxIMax(cbox));

      /* coarsen the grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
   }
   num_levels = l + 1;

   /* free up some things */
   hypre_BoxDestroy(cbox);

   (cyc_red_data -> num_levels)      = num_levels;
   (cyc_red_data -> grid_l)          = grid_l;

   /*-----------------------------------------------------
    * Set up base points
    *-----------------------------------------------------*/

   base_points = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[0]));
   hypre_ProjectBoxArray(base_points, base_index, base_stride);

   (cyc_red_data -> base_points) = base_points;

   /*-----------------------------------------------------
    * Set up fine points
    *-----------------------------------------------------*/

   fine_points_l   = hypre_TAlloc(hypre_BoxArray *,  num_levels);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      fine_points_l[l] =
         hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[l]));
      hypre_ProjectBoxArray(fine_points_l[l], findex, stride);
   }
  
   fine_points_l[l] =
      hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid_l[l]));
   if (num_levels == 1)
   {
      hypre_ProjectBoxArray(fine_points_l[l], base_index, base_stride);
   }

   (cyc_red_data -> fine_points_l)   = fine_points_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);

   A_l[0] = hypre_StructMatrixRef(A);
   x_l[0] = hypre_StructVectorRef(x);

   x_num_ghost[2*cdir]     = 1;
   x_num_ghost[2*cdir + 1] = 1;

   for (l = 0; l < (num_levels - 1); l++)
   {
      A_l[l+1] = hypre_CycRedCreateCoarseOp(A_l[l], grid_l[l+1], cdir);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);

      x_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(x_l[l+1], x_num_ghost);
      hypre_StructVectorInitializeShell(x_l[l+1]);
      data_size += hypre_StructVectorDataSize(x_l[l+1]);
   }

   data = hypre_SharedCTAlloc(double, data_size);

   (cyc_red_data -> data) = data;

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_StructMatrixInitializeData(A_l[l+1], data);
      data += hypre_StructMatrixDataSize(A_l[l+1]);
      hypre_StructVectorInitializeData(x_l[l+1], data);
      hypre_StructVectorAssemble(x_l[l+1]);
      data += hypre_StructVectorDataSize(x_l[l+1]);
   }

   (cyc_red_data -> A_l)  = A_l;
   (cyc_red_data -> x_l)  = x_l;

   /*-----------------------------------------------------
    * Set up coarse grid operators
    *-----------------------------------------------------*/

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      hypre_CycRedSetupCoarseOp(A_l[l], A_l[l+1], cindex, stride);
   }

   /*----------------------------------------------------------
    * Set up compute packages
    *----------------------------------------------------------*/

   down_compute_pkg_l = hypre_TAlloc(hypre_ComputePkg *, (num_levels - 1));
   up_compute_pkg_l   = hypre_TAlloc(hypre_ComputePkg *, (num_levels - 1));

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      hypre_CreateComputeInfo(grid_l[l], hypre_StructMatrixStencil(A_l[l]),
                              &send_boxes, &recv_boxes,
                              &send_processes, &recv_processes,
                              &indt_boxes, &dept_boxes);
 
      /* down-cycle */
      hypre_ProjectBoxArrayArray(send_boxes, findex, stride);
      hypre_ProjectBoxArrayArray(recv_boxes, findex, stride);
      hypre_ProjectBoxArrayArray(indt_boxes, cindex, stride);
      hypre_ProjectBoxArrayArray(dept_boxes, cindex, stride);
      hypre_ComputePkgCreate(send_boxes, recv_boxes,
                             stride, stride,
                             send_processes, recv_processes,
                             indt_boxes, dept_boxes,
                             stride, grid_l[l],
                             hypre_StructVectorDataSpace(x_l[l]), 1,
                             &down_compute_pkg_l[l]);

      hypre_CreateComputeInfo(grid_l[l], hypre_StructMatrixStencil(A_l[l]),
                              &send_boxes, &recv_boxes,
                              &send_processes, &recv_processes,
                              &indt_boxes, &dept_boxes);

      /* up-cycle */
      hypre_ProjectBoxArrayArray(send_boxes, cindex, stride);
      hypre_ProjectBoxArrayArray(recv_boxes, cindex, stride);
      hypre_ProjectBoxArrayArray(indt_boxes, findex, stride);
      hypre_ProjectBoxArrayArray(dept_boxes, findex, stride);
      hypre_ComputePkgCreate(send_boxes, recv_boxes,
                             stride, stride,
                             send_processes, recv_processes,
                             indt_boxes, dept_boxes,
                             stride, grid_l[l],
                             hypre_StructVectorDataSpace(x_l[l]), 1,
                             &up_compute_pkg_l[l]);
   }

   (cyc_red_data -> down_compute_pkg_l) = down_compute_pkg_l;
   (cyc_red_data -> up_compute_pkg_l)   = up_compute_pkg_l;

   /*-----------------------------------------------------
    * Compute solve flops
    *-----------------------------------------------------*/

   flop_divisor = (hypre_IndexX(base_stride) *
                   hypre_IndexY(base_stride) *
                   hypre_IndexZ(base_stride)  );
   (cyc_red_data -> solve_flops) =
      hypre_StructVectorGlobalSize(x_l[0])/2/flop_divisor;
   (cyc_red_data -> solve_flops) +=
      5*hypre_StructVectorGlobalSize(x_l[0])/2/flop_divisor;
   for (l = 1; l < (num_levels - 1); l++)
   {
      (cyc_red_data -> solve_flops) +=
         10*hypre_StructVectorGlobalSize(x_l[l])/2;
   }

   if (num_levels > 1)
   {
      (cyc_red_data -> solve_flops) +=  
          hypre_StructVectorGlobalSize(x_l[l])/2;
   }
   

   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

#if DEBUG
   {
      char  filename[255];

      /* debugging stuff */
      for (l = 0; l < num_levels; l++)
      {
         sprintf(filename, "yout_A.%02d", l);
         hypre_StructMatrixPrint(filename, A_l[l], 0);
      }
   }
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReduction
 *
 * The solution vectors on each level are also used to store the
 * right-hand-side data.  We can do this because of the red-black
 * nature of the algorithm and the fact that the method is exact,
 * allowing one to assume initial guesses of zero on all grid levels.
 *--------------------------------------------------------------------------*/

int
hypre_CyclicReduction( void               *cyc_red_vdata,
                       hypre_StructMatrix *A,
                       hypre_StructVector *b,
                       hypre_StructVector *x             )
{
   hypre_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   int                   num_levels      = (cyc_red_data -> num_levels);
   int                   cdir            = (cyc_red_data -> cdir);
   hypre_IndexRef        base_index      = (cyc_red_data -> base_index);
   hypre_IndexRef        base_stride     = (cyc_red_data -> base_stride);
   hypre_BoxArray       *base_points     = (cyc_red_data -> base_points);
   hypre_BoxArray      **fine_points_l   = (cyc_red_data -> fine_points_l);
   hypre_StructMatrix  **A_l             = (cyc_red_data -> A_l);
   hypre_StructVector  **x_l             = (cyc_red_data -> x_l);
   hypre_ComputePkg    **down_compute_pkg_l =
      (cyc_red_data -> down_compute_pkg_l);
   hypre_ComputePkg    **up_compute_pkg_l   =
      (cyc_red_data -> up_compute_pkg_l);
                    
   hypre_StructGrid     *fgrid;
   int                  *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   int                  *cgrid_ids;

   hypre_CommHandle     *comm_handle;
                     
   hypre_BoxArrayArray  *compute_box_aa;
   hypre_BoxArray       *compute_box_a;
   hypre_Box            *compute_box;
                     
   hypre_Box            *A_dbox;
   hypre_Box            *x_dbox;
   hypre_Box            *b_dbox;
   hypre_Box            *xc_dbox;
                     
   double               *Ap, *Awp, *Aep;
   double               *xp, *xwp, *xep;
   double               *bp;
   double               *xcp;
                       
   int                   Ai;
   int                   xi;
   int                   bi;
   int                   xci;
                     
   hypre_Index           cindex;
   hypre_Index           stride;
                       
   hypre_Index           index;
   hypre_Index           loop_size;
   hypre_Index           start;
   hypre_Index           startc;
   hypre_Index           stridec;
                     
   int                   compute_i, fi, ci, j, l;
   int                   loopi, loopj, loopk;
                      
   int                   ierr = 0;

   hypre_BeginTiming(cyc_red_data -> time_index);


   /*--------------------------------------------------
    * Initialize some things
    *--------------------------------------------------*/

   hypre_SetIndex(stridec, 1, 1, 1);

   hypre_StructMatrixDestroy(A_l[0]);
   hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   x_l[0] = hypre_StructVectorRef(x);

   /*--------------------------------------------------
    * Copy b into x
    *--------------------------------------------------*/

   compute_box_a = base_points;
   hypre_ForBoxI(fi, compute_box_a)
      {
         compute_box = hypre_BoxArrayBox(compute_box_a, fi);

         x_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), fi);
         b_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), fi);

         xp = hypre_StructVectorBoxData(x, fi);
         bp = hypre_StructVectorBoxData(b, fi);

         hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
         hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                                x_dbox, start, base_stride, xi,
                                b_dbox, start, base_stride, bi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,bi
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, xi, bi)
            {
               xp[xi] = bp[bi];
            }
         hypre_BoxLoop2End(xi, bi);
      }

   /*--------------------------------------------------
    * Down cycle:
    *
    * 1) Do an F-relaxation sweep with zero initial guess
    * 2) Compute and inject residual at C-points
    *    - computations are at C-points
    *    - communications are at F-points
    *
    * Notes:
    * - Before these two steps are executed, the
    * fine-grid solution vector contains the right-hand-side.
    * - After these two steps are executed, the fine-grid
    * solution vector contains the right-hand side at
    * C-points and the current solution approximation at
    * F-points.  The coarse-grid solution vector contains
    * the restricted (injected) fine-grid residual.
    * - The coarsest grid solve is built into this loop
    * because it involves the same code as step 1.
    *--------------------------------------------------*/
 
   /* The break out of this loop is just before step 2 below */
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      compute_box_a = fine_points_l[l];
      hypre_ForBoxI(fi, compute_box_a)
         {
            compute_box = hypre_BoxArrayBox(compute_box_a, fi);

            A_dbox =
               hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
            x_dbox =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);

            hypre_SetIndex(index, 0, 0, 0);
            Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xp = hypre_StructVectorBoxData(x_l[l], fi);

            hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
            hypre_BoxGetStrideSize(compute_box, stride, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                A_dbox, start, stride, Ai,
                                x_dbox, start, stride, xi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, Ai, xi)
               {
                  xp[xi] /= Ap[Ai]; 
               }
            hypre_BoxLoop2End(Ai, xi);
         }

      if (l == (num_levels - 1))
         break;

      /* Step 2 */
      fgrid = hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = hypre_StructGridIDs(fgrid);
      cgrid = hypre_StructVectorGrid(x_l[l+1]);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      cgrid_ids = hypre_StructGridIDs(cgrid);
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               hypre_InitializeIndtComputations(down_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa =
                  hypre_ComputePkgIndtBoxes(down_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa =
                  hypre_ComputePkgDeptBoxes(down_compute_pkg_l[l]);
            }
            break;
         }

         fi = 0;
         hypre_ForBoxArrayI(ci, cgrid_boxes)
            {
               while (fgrid_ids[fi] != cgrid_ids[ci])
               {
                  fi++;
               }

               compute_box_a =
                  hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

               A_dbox =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
               x_dbox =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);
               xc_dbox =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l+1]), ci);

               xp  = hypre_StructVectorBoxData(x_l[l], fi);
               xcp = hypre_StructVectorBoxData(x_l[l+1], ci);

               hypre_SetIndex(index, -1, 0, 0);
               Awp =
                  hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
               xwp = hypre_StructVectorBoxData(x_l[l], fi) +
                  hypre_BoxOffsetDistance(x_dbox, index);

               hypre_SetIndex(index,  1, 0, 0);
               Aep =
                  hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
               xep = hypre_StructVectorBoxData(x_l[l], fi) +
                  hypre_BoxOffsetDistance(x_dbox, index);

               hypre_ForBoxI(j, compute_box_a)
                  {
                     compute_box = hypre_BoxArrayBox(compute_box_a, j);

                     hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
                     hypre_StructMapFineToCoarse(start, cindex, stride,
                                                 startc);

                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                     hypre_BoxLoop3Begin(loop_size,
                                         A_dbox, start, stride, Ai,
                                         x_dbox, start, stride, xi,
                                         xc_dbox, startc, stridec, xci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,xci
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, xci)
                        {
                           xcp[xci] = xp[xi] -
                              Awp[Ai]*xwp[xi] -
                              Aep[Ai]*xep[xi];
                        }
                     hypre_BoxLoop3End(Ai, xi, xci);
                  }
            }
      }
   }

   /*--------------------------------------------------
    * Up cycle:
    *
    * 1) Inject coarse error into fine-grid solution
    *    vector (this is the solution at the C-points)
    * 2) Do an F-relaxation sweep on Ax = 0 and update
    *    solution at F-points
    *    - computations are at F-points
    *    - communications are at C-points
    *--------------------------------------------------*/

   for (l = (num_levels - 2); l >= 0; l--)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      fgrid = hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = hypre_StructGridIDs(fgrid);
      cgrid = hypre_StructVectorGrid(x_l[l+1]);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      cgrid_ids = hypre_StructGridIDs(cgrid);
      fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
         {
            while (fgrid_ids[fi] != cgrid_ids[ci])
            {
               fi++;
            }

            compute_box = hypre_BoxArrayBox(cgrid_boxes, ci);

            hypre_CopyIndex(hypre_BoxIMin(compute_box), startc);
            hypre_StructMapCoarseToFine(startc, cindex, stride, start);

            x_dbox =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);
            xc_dbox =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l+1]), ci);

            xp  = hypre_StructVectorBoxData(x_l[l], fi);
            xcp = hypre_StructVectorBoxData(x_l[l+1], ci);

            hypre_BoxGetSize(compute_box, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                x_dbox, start, stride, xi,
                                xc_dbox, startc, stridec, xci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,xci
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, xi, xci)
               {
                  xp[xi] = xcp[xci];
               }
            hypre_BoxLoop2End(xi, xci);
         }

      /* Step 2 */
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               hypre_InitializeIndtComputations(up_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa =
                  hypre_ComputePkgIndtBoxes(up_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa =
                  hypre_ComputePkgDeptBoxes(up_compute_pkg_l[l]);
            }
            break;
         }

         hypre_ForBoxArrayI(fi, compute_box_aa)
            {
               compute_box_a =
                  hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

               A_dbox =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), fi);
               x_dbox =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), fi);

               hypre_SetIndex(index, 0, 0, 0);
               Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
               xp = hypre_StructVectorBoxData(x_l[l], fi);

               hypre_SetIndex(index, -1, 0, 0);
               Awp =
                  hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
               xwp = hypre_StructVectorBoxData(x_l[l], fi) +
                  hypre_BoxOffsetDistance(x_dbox, index);

               hypre_SetIndex(index,  1, 0, 0);
               Aep =
                  hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
               xep = hypre_StructVectorBoxData(x_l[l], fi) +
                  hypre_BoxOffsetDistance(x_dbox, index);

               hypre_ForBoxI(j, compute_box_a)
                  {
                     compute_box = hypre_BoxArrayBox(compute_box_a, j);

                     hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                     hypre_BoxLoop2Begin(loop_size,
                                         A_dbox, start, stride, Ai,
                                         x_dbox, start, stride, xi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop2For(loopi, loopj, loopk, Ai, xi)
                        {
                           xp[xi] -= (Awp[Ai]*xwp[xi] +
                                      Aep[Ai]*xep[xi]  ) / Ap[Ai];
                        }
                     hypre_BoxLoop2End(Ai, xi);
                  }
            }
      }
   }

   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

   hypre_IncFLOPCount(cyc_red_data -> solve_flops);
   hypre_EndTiming(cyc_red_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionSetBase
 *--------------------------------------------------------------------------*/
 
int
hypre_CyclicReductionSetBase( void        *cyc_red_vdata,
                              hypre_Index  base_index,
                              hypre_Index  base_stride )
{
   hypre_CyclicReductionData *cyc_red_data = cyc_red_vdata;
   int                      d;
   int                      ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((cyc_red_data -> base_index),  d) =
         hypre_IndexD(base_index,  d);
      hypre_IndexD((cyc_red_data -> base_stride), d) =
         hypre_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionDestroy
 *--------------------------------------------------------------------------*/

int
hypre_CyclicReductionDestroy( void *cyc_red_vdata )
{
   hypre_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   int l;
   int ierr = 0;

   if (cyc_red_data)
   {
      hypre_BoxArrayDestroy(cyc_red_data -> base_points);
      hypre_StructGridDestroy(cyc_red_data -> grid_l[0]);
      hypre_StructMatrixDestroy(cyc_red_data -> A_l[0]);
      hypre_StructVectorDestroy(cyc_red_data -> x_l[0]);
      for (l = 0; l < ((cyc_red_data -> num_levels) - 1); l++)
      {
         hypre_StructGridDestroy(cyc_red_data -> grid_l[l+1]);
         hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
         hypre_StructMatrixDestroy(cyc_red_data -> A_l[l+1]);
         hypre_StructVectorDestroy(cyc_red_data -> x_l[l+1]);
         hypre_ComputePkgDestroy(cyc_red_data -> down_compute_pkg_l[l]);
         hypre_ComputePkgDestroy(cyc_red_data -> up_compute_pkg_l[l]);
      }
      hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
      hypre_SharedTFree(cyc_red_data -> data); 
      hypre_TFree(cyc_red_data -> grid_l);
      hypre_TFree(cyc_red_data -> fine_points_l);
      hypre_TFree(cyc_red_data -> A_l);
      hypre_TFree(cyc_red_data -> x_l);
      hypre_TFree(cyc_red_data -> down_compute_pkg_l);
      hypre_TFree(cyc_red_data -> up_compute_pkg_l);

      hypre_FinalizeTiming(cyc_red_data -> time_index);
      hypre_TFree(cyc_red_data);
   }

   return ierr;
}


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

/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define hypre_CycRedMapFineToCoarse(index1, index2, cindex, cstride) \
{\
   hypre_IndexX(index2) =\
      (hypre_IndexX(index1) - hypre_IndexX(cindex)) / hypre_IndexX(cstride);\
   hypre_IndexY(index2) =\
      (hypre_IndexY(index1) - hypre_IndexY(cindex)) / hypre_IndexY(cstride);\
   hypre_IndexZ(index2) =\
      (hypre_IndexZ(index1) - hypre_IndexZ(cindex)) / hypre_IndexZ(cstride);\
}
 
#define hypre_CycRedMapCoarseToFine(index1, index2, cindex, cstride) \
{\
   hypre_IndexX(index2) =\
      hypre_IndexX(index1) * hypre_IndexX(cstride) + hypre_IndexX(cindex);\
   hypre_IndexY(index2) =\
      hypre_IndexY(index1) * hypre_IndexY(cstride) + hypre_IndexY(cindex);\
   hypre_IndexZ(index2) =\
      hypre_IndexZ(index1) * hypre_IndexZ(cstride) + hypre_IndexZ(cindex);\
}

#define hypre_CycRedSetCIndex(base_index, base_stride, level, cdir, cindex) \
{\
   if (level > 0)\
      hypre_SetIndex(cindex, 0, 0, 0);\
   else\
      hypre_CopyIndex(base_index,  cindex);\
   hypre_IndexD(cindex, cdir) = 0;\
}

#define hypre_CycRedSetFIndex(base_index, base_stride, level, cdir, findex) \
{\
   if (level > 0)\
      hypre_SetIndex(findex, 0, 0, 0);\
   else\
      hypre_CopyIndex(base_index,  findex);\
   hypre_IndexD(findex, cdir) = 1;\
}

#define hypre_CycRedSetStride(base_index, base_stride, level, cdir, stride) \
{\
   if (level > 0)\
      hypre_SetIndex(stride, 1, 1, 1);\
   else\
      hypre_CopyIndex(base_stride, stride);\
   hypre_IndexD(stride, cdir) = 2;\
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
                    
   hypre_SBoxArray      *base_points;
   hypre_SBoxArray     **fine_points_l;
   hypre_SBoxArray     **coarse_points_l;

   double               *data;
   hypre_StructMatrix  **A_l;
   hypre_StructVector  **x_l;

   hypre_ComputePkg    **down_compute_pkg_l;
   hypre_ComputePkg    **up_compute_pkg_l;

   int                   time_index;
   int                   solve_flops;

} hypre_CyclicReductionData;

/*--------------------------------------------------------------------------
 * hypre_CyclicReductionInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_CyclicReductionInitialize( MPI_Comm  comm )
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
 * hypre_CycRedNewCoarseOp
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_CycRedNewCoarseOp( hypre_StructMatrix *A,
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

   Ac_stencil = hypre_NewStructStencil(Ac_stencil_dim, Ac_stencil_size,
                                       Ac_stencil_shape);

   Ac = hypre_NewStructMatrix(hypre_StructMatrixComm(A),
                              coarse_grid, Ac_stencil);

   /*-----------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------*/

   hypre_StructMatrixSymmetric(Ac) = hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------
    * Set number of ghost points
    *-----------------------------------------------*/

   Ac_num_ghost[2*cdir]     = 1;
   Ac_num_ghost[2*cdir + 1] = 1;
   hypre_SetStructMatrixNumGhost(Ac, Ac_num_ghost);

   hypre_InitializeStructMatrixShell(Ac);
 
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
   hypre_Index             index_temp;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             fstart;
   hypre_IndexRef          stridef;
   hypre_Index             loop_size;

   int                     i;
   int                     loopi, loopj, loopk;

   hypre_Box              *A_data_box;
   hypre_Box              *Ac_data_box;

   double                 *a_cc, *a_cw, *a_ce;
   double                 *ac_cc, *ac_cw, *ac_ce;
                    
   int                     iA, iAm1, iAp1;
   int                     iAc;
                         
   int                     xOffsetA; 
                         
   int                     ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(Ac);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   hypre_ForBoxI(i, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, i);

         cstart = hypre_BoxIMin(cgrid_box);
         hypre_CycRedMapCoarseToFine(cstart, fstart, cindex, cstride) ;

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         Ac_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Ac), i);

         /*-----------------------------------------------
          * Extract pointers for 3-point fine grid operator:
          * 
          * a_cc is pointer for center coefficient
          * a_cw is pointer for west coefficient
          * a_ce is pointer for east coefficient
          *-----------------------------------------------*/

         hypre_SetIndex(index_temp,0,0,0);
         a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index_temp);

         hypre_SetIndex(index_temp,-1,0,0);
         a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index_temp);

         hypre_SetIndex(index_temp,1,0,0);
         a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index_temp);

         /*-----------------------------------------------
          * Extract pointers for coarse grid operator - always 3-point:
          *
          * If A is symmetric so is Ac.  We build only the
          * lower triangular part (plus diagonal).
          * 
          * ac_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------*/

         hypre_SetIndex(index_temp,0,0,0);
         ac_cc = hypre_StructMatrixExtractPointerByIndex(Ac, i, index_temp);

         hypre_SetIndex(index_temp,-1,0,0);
         ac_cw = hypre_StructMatrixExtractPointerByIndex(Ac, i, index_temp);

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_SetIndex(index_temp,1,0,0);
            ac_ce = hypre_StructMatrixExtractPointerByIndex(Ac, i, index_temp);
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

         hypre_SetIndex(index_temp,1,0,0);
         xOffsetA = hypre_BoxOffsetDistance(A_data_box,index_temp); 

         /*-----------------------------------------------
          * non-symmetric case
          *-----------------------------------------------*/

         if(!hypre_StructMatrixSymmetric(A))
         {
            hypre_GetBoxSize(cgrid_box, loop_size);
            hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                           A_data_box, fstart, stridef, iA,
                           Ac_data_box, cstart, stridec, iAc,
                           {
                              iAm1 = iA - xOffsetA;
                              iAp1 = iA + xOffsetA;

                              ac_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                              ac_cc[iAc] = a_cc[iA]
                                 - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                                 - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   

                              ac_ce[iAc] = - a_ce[iA] *a_ce[iAp1] / a_cc[iAp1];

                           });
         }

         /*-----------------------------------------------
          * symmetric case
          *-----------------------------------------------*/

         else
         {
            hypre_GetBoxSize(cgrid_box, loop_size);
            hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                           A_data_box, fstart, stridef, iA,
                           Ac_data_box, cstart, stridec, iAc,
                           {
                              iAm1 = iA - xOffsetA;
                              iAp1 = iA + xOffsetA;

                              ac_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                              ac_cc[iAc] = a_cc[iA]
                                 - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                                 - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   
                           });
         }

      } /* end ForBoxI */

   hypre_AssembleStructMatrix(Ac);

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
   hypre_SBoxArray        *base_points;
   hypre_SBoxArray       **fine_points_l;
   hypre_SBoxArray       **coarse_points_l;
   double                 *data;
   int                     data_size = 0;
   hypre_StructMatrix    **A_l;
   hypre_StructVector    **x_l;
   hypre_ComputePkg      **down_compute_pkg_l;
   hypre_ComputePkg      **up_compute_pkg_l;

   hypre_BoxArray         *coarsest_boxes;

   hypre_Index             cindex;
   hypre_Index             findex;
   hypre_Index             stride;

   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_SBoxArrayArray   *send_sboxes;
   hypre_SBoxArrayArray   *recv_sboxes;
   hypre_SBoxArrayArray   *indt_sboxes;
   hypre_SBoxArrayArray   *dept_sboxes;

   hypre_BoxArray         *boxes;
   hypre_BoxArray         *all_boxes;
   int                    *processes;
   int                    *box_ranks;
   hypre_SBoxArray        *coarse_points;
   int                     num_boxes;
   int                     num_all_boxes;
 
   hypre_SBox             *sbox;
   hypre_Box              *box;
                    
   int                     idmin, idmax;
   int                     i, l;
   int                     flop_divisor;
                         
   int                     x_num_ghost[] = {0, 0, 0, 0, 0, 0};
                         
   int                     ierr = 0;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_GatherAllBoxes(comm, boxes, &all_boxes, &processes, &box_ranks);
   num_boxes = hypre_BoxArraySize(boxes);
   num_all_boxes = hypre_BoxArraySize(all_boxes);

   /* Compute a preliminary num_levels value based on the grid */
   idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   for (i = 0; i < num_all_boxes; i++)
   {
      idmin =
         min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
      idmax =
         max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
   }
   num_levels = hypre_Log2(idmax - idmin + 1) + 2;

   grid_l    = hypre_TAlloc(hypre_StructGrid *, num_levels);
   grid_l[0] = hypre_StructMatrixGrid(A);
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      for (i = 0; i < num_all_boxes; i++)
      {
         idmin =
            min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
         idmax =
            max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
      }
      if ( idmin == idmax )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen the grid by coarsening all_boxes (reduces communication) */
      coarse_points = hypre_ProjectBoxArray(all_boxes, cindex, stride);
      hypre_FreeBoxArray(all_boxes);
      all_boxes = hypre_NewBoxArray(num_all_boxes);
      for (i = 0; i < num_all_boxes; i++)
      {
         sbox = hypre_SBoxArraySBox(coarse_points, i);
         box = hypre_DuplicateBox(hypre_SBoxBox(sbox));
         hypre_CycRedMapFineToCoarse(hypre_BoxIMin(box), hypre_BoxIMin(box),
                                     cindex, stride);
         hypre_CycRedMapFineToCoarse(hypre_BoxIMax(box), hypre_BoxIMax(box),
                                     cindex, stride);
         hypre_AppendBox(box, all_boxes);
      }
      hypre_FreeSBoxArray(coarse_points);

      /* compute local boxes */
      boxes = hypre_NewBoxArray(num_boxes);
      for (i = 0; i < num_boxes; i++)
      {
         box = hypre_DuplicateBox(hypre_BoxArrayBox(all_boxes, box_ranks[i]));
         hypre_AppendBox(box, boxes);
      }

      grid_l[l+1] = hypre_NewStructGrid(comm, hypre_StructGridDim(grid_l[l]));
      hypre_SetStructGridBoxes(grid_l[l+1], boxes);
      hypre_AssembleStructGrid(grid_l[l+1], all_boxes, processes, box_ranks);
   }
   num_levels = l + 1;

   hypre_FreeBoxArray(all_boxes);
   hypre_TFree(processes);
   hypre_TFree(box_ranks);

   (cyc_red_data -> num_levels)      = num_levels;
   (cyc_red_data -> grid_l)          = grid_l;

   /*-----------------------------------------------------
    * Set up base points
    *-----------------------------------------------------*/

   base_points = hypre_ProjectBoxArray(hypre_StructGridBoxes(grid_l[0]),
                                       base_index, base_stride);

   (cyc_red_data -> base_points) = base_points;

   /*-----------------------------------------------------
    * Set up fine and coarse points
    *-----------------------------------------------------*/

   fine_points_l   = hypre_TAlloc(hypre_SBoxArray *,  num_levels);
   coarse_points_l = hypre_TAlloc(hypre_SBoxArray *,  num_levels - 1);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      fine_points_l[l]   =
         hypre_ProjectBoxArray(hypre_StructGridBoxes(grid_l[l]),
                               findex, stride);
      coarse_points_l[l] =
         hypre_ProjectBoxArray(hypre_StructGridBoxes(grid_l[l]),
                               cindex, stride);
   }
  
   if (num_levels == 1)
   {
      fine_points_l[l]   =
         hypre_ProjectBoxArray(hypre_StructGridBoxes(grid_l[l]),
                               base_index, base_stride);
   }
   else
   {
      coarsest_boxes = 
         hypre_DuplicateBoxArray(hypre_StructGridBoxes(grid_l[l]));
      fine_points_l[l] = hypre_ConvertToSBoxArray(coarsest_boxes);
   }

   (cyc_red_data -> fine_points_l)   = fine_points_l;
   (cyc_red_data -> coarse_points_l) = coarse_points_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);

   A_l[0] = A;
   x_l[0] = x;

   x_num_ghost[2*cdir]     = 1;
   x_num_ghost[2*cdir + 1] = 1;

   for (l = 0; l < (num_levels - 1); l++)
   {
      A_l[l+1] = hypre_CycRedNewCoarseOp(A_l[l], grid_l[l+1], cdir);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);

      x_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      hypre_InitializeStructVectorShell(x_l[l+1]);
      data_size += hypre_StructVectorDataSize(x_l[l+1]);
   }

   data = hypre_CTAlloc(double, data_size);
   (cyc_red_data -> data) = data;

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_InitializeStructMatrixData(A_l[l+1], data);
      data += hypre_StructMatrixDataSize(A_l[l+1]);

      hypre_InitializeStructVectorData(x_l[l+1], data);
      hypre_AssembleStructVector(x_l[l+1]);
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

      hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                           &send_processes, &recv_processes,
                           &indt_boxes, &dept_boxes,
                           grid_l[l], hypre_StructMatrixStencil(A_l[l]));
 
      /* down-cycle */
      send_sboxes = hypre_ProjectBoxArrayArray(send_boxes, findex, stride);
      recv_sboxes = hypre_ProjectBoxArrayArray(recv_boxes, findex, stride);
      indt_sboxes = hypre_ProjectBoxArrayArray(indt_boxes, cindex, stride);
      dept_sboxes = hypre_ProjectBoxArrayArray(dept_boxes, cindex, stride);
      down_compute_pkg_l[l] =
         hypre_NewComputePkg(send_sboxes, recv_sboxes,
                             send_processes, recv_processes,
                             indt_sboxes, dept_sboxes,
                             grid_l[l],
                             hypre_StructVectorDataSpace(x_l[l]), 1);

      hypre_FreeBoxArrayArray(send_boxes);
      hypre_FreeBoxArrayArray(recv_boxes);
      hypre_FreeBoxArrayArray(indt_boxes);
      hypre_FreeBoxArrayArray(dept_boxes);

      hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                           &send_processes, &recv_processes,
                           &indt_boxes, &dept_boxes,
                           grid_l[l], hypre_StructMatrixStencil(A_l[l]));

      /* up-cycle */
      send_sboxes = hypre_ProjectBoxArrayArray(send_boxes, cindex, stride);
      recv_sboxes = hypre_ProjectBoxArrayArray(recv_boxes, cindex, stride);
      indt_sboxes = hypre_ProjectBoxArrayArray(indt_boxes, findex, stride);
      dept_sboxes = hypre_ProjectBoxArrayArray(dept_boxes, findex, stride);
      up_compute_pkg_l[l] =
         hypre_NewComputePkg(send_sboxes, recv_sboxes,
                             send_processes, recv_processes,
                             indt_sboxes, dept_sboxes,
                             grid_l[l],
                             hypre_StructVectorDataSpace(x_l[l]), 1);

      hypre_FreeBoxArrayArray(send_boxes);
      hypre_FreeBoxArrayArray(recv_boxes);
      hypre_FreeBoxArrayArray(indt_boxes);
      hypre_FreeBoxArrayArray(dept_boxes);
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

#if 0
   {
      char  filename[255];

      /* debugging stuff */
      for (l = 0; l < num_levels; l++)
      {
         sprintf(filename, "yout_A.%02d", l);
         hypre_PrintStructMatrix(filename, A_l[l], 0);
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

   int                   num_levels         = (cyc_red_data -> num_levels);
   int                   cdir               = (cyc_red_data -> cdir);
   hypre_IndexRef        base_index         = (cyc_red_data -> base_index);
   hypre_IndexRef        base_stride        = (cyc_red_data -> base_stride);
   hypre_SBoxArray      *base_points        = (cyc_red_data -> base_points);
   hypre_SBoxArray     **fine_points_l      = (cyc_red_data -> fine_points_l);
   hypre_SBoxArray     **coarse_points_l    = (cyc_red_data -> coarse_points_l);
   hypre_StructMatrix  **A_l                = (cyc_red_data -> A_l);
   hypre_StructVector  **x_l                = (cyc_red_data -> x_l);
   hypre_ComputePkg    **down_compute_pkg_l =
      (cyc_red_data -> down_compute_pkg_l);
   hypre_ComputePkg    **up_compute_pkg_l   =
      (cyc_red_data -> up_compute_pkg_l);
                    
   hypre_CommHandle     *comm_handle;
                     
   hypre_SBoxArrayArray *compute_sbox_aa;
   hypre_SBoxArray      *compute_sbox_a;
   hypre_SBox           *compute_sbox;
                     
   hypre_Box            *A_data_box;
   hypre_Box            *x_data_box;
   hypre_Box            *b_data_box;
   hypre_Box            *xc_data_box;
                     
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
   hypre_IndexRef        start;
   hypre_Index           startc;
   hypre_Index           stridec;
                     
   int                   compute_i, i, j, l;
   int                   loopi, loopj, loopk;
                      
   int                   ierr = 0;

   hypre_BeginTiming(cyc_red_data -> time_index);

   /*--------------------------------------------------
    * Initialize some things
    *--------------------------------------------------*/

   hypre_SetIndex(stridec, 1, 1, 1);

   A_l[0] = A;
   x_l[0] = x;

   /*--------------------------------------------------
    * Copy b into x
    *--------------------------------------------------*/

   compute_sbox_a = base_points;
   hypre_ForSBoxI(i, compute_sbox_a)
      {
         compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);

         x_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         b_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);

         xp = hypre_StructVectorBoxData(x, i);
         bp = hypre_StructVectorBoxData(b, i);

         start  = hypre_SBoxIMin(compute_sbox);

         hypre_GetSBoxSize(compute_sbox, loop_size);
         hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                        x_data_box, start, base_stride, xi,
                        b_data_box, start, base_stride, bi,
                        {
                           xp[xi] = bp[bi];
                        });
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
      compute_sbox_a = fine_points_l[l];
      hypre_ForSBoxI(i, compute_sbox_a)
         {
            compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);

            A_data_box =
               hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), i);
            x_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), i);

            hypre_SetIndex(index, 0, 0, 0);
            Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xp = hypre_StructVectorBoxData(x_l[l], i);

            start  = hypre_SBoxIMin(compute_sbox);

            hypre_GetSBoxSize(compute_sbox, loop_size);
            hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                           A_data_box,  start,  stride,  Ai,
                           x_data_box,  start,  stride,  xi,
                           {
                              xp[xi] /= Ap[Ai];
                           });
         }

      if (l == (num_levels - 1))
         break;

      /* Step 2 */
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               comm_handle =
                  hypre_InitializeIndtComputations(down_compute_pkg_l[l], xp);
               compute_sbox_aa =
                  hypre_ComputePkgIndtSBoxes(down_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_sbox_aa =
                  hypre_ComputePkgDeptSBoxes(down_compute_pkg_l[l]);
            }
            break;
         }

         hypre_ForSBoxArrayI(i, compute_sbox_aa)
            {
               compute_sbox_a =
                  hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

               A_data_box =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), i);
               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), i);
               xc_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l+1]), i);

               xp  = hypre_StructVectorBoxData(x_l[l], i);
               xcp = hypre_StructVectorBoxData(x_l[l+1], i);

               hypre_SetIndex(index, -1, 0, 0);
               Awp = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
               xwp = hypre_StructVectorBoxData(x_l[l], i) +
                  hypre_BoxOffsetDistance(x_data_box, index);

               hypre_SetIndex(index,  1, 0, 0);
               Aep = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
               xep = hypre_StructVectorBoxData(x_l[l], i) +
                  hypre_BoxOffsetDistance(x_data_box, index);

               hypre_ForSBoxI(j, compute_sbox_a)
                  {
                     compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

                     start  = hypre_SBoxIMin(compute_sbox);
                     hypre_CycRedMapFineToCoarse(start, startc,
                                                 cindex, stride);

                     hypre_GetSBoxSize(compute_sbox, loop_size);
                     hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
                                    A_data_box,  start,  stride,  Ai,
                                    x_data_box,  start,  stride,  xi,
                                    xc_data_box, startc, stridec, xci,
                                    {
                                       xcp[xci] = xp[xi] -
                                          Awp[Ai]*xwp[xi] -
                                          Aep[Ai]*xep[xi];
                                    });
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
      compute_sbox_a = coarse_points_l[l];
      hypre_ForSBoxI(i, compute_sbox_a)
         {
            compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);

            x_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), i);
            xc_data_box =
               hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l+1]), i);

            xp  = hypre_StructVectorBoxData(x_l[l], i);
            xcp = hypre_StructVectorBoxData(x_l[l+1], i);

            start  = hypre_SBoxIMin(compute_sbox);
            hypre_CycRedMapFineToCoarse(start, startc, cindex, stride);

            hypre_GetSBoxSize(compute_sbox, loop_size);
            hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                           x_data_box,  start,  stride,  xi,
                           xc_data_box, startc, stridec, xci,
                           {
                              xp[xi] = xcp[xci];
                           });
         }

      /* Step 2 */
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x_l[l]);
               comm_handle =
                  hypre_InitializeIndtComputations(up_compute_pkg_l[l], xp);
               compute_sbox_aa =
                  hypre_ComputePkgIndtSBoxes(up_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_sbox_aa =
                  hypre_ComputePkgDeptSBoxes(up_compute_pkg_l[l]);
            }
            break;
         }

         hypre_ForSBoxArrayI(i, compute_sbox_aa)
            {
               compute_sbox_a =
                  hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

               A_data_box =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_l[l]), i);
               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x_l[l]), i);

               hypre_SetIndex(index, 0, 0, 0);
               Ap = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
               xp = hypre_StructVectorBoxData(x_l[l], i);

               hypre_SetIndex(index, -1, 0, 0);
               Awp = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
               xwp = hypre_StructVectorBoxData(x_l[l], i) +
                  hypre_BoxOffsetDistance(x_data_box, index);

               hypre_SetIndex(index,  1, 0, 0);
               Aep = hypre_StructMatrixExtractPointerByIndex(A_l[l], i, index);
               xep = hypre_StructVectorBoxData(x_l[l], i) +
                  hypre_BoxOffsetDistance(x_data_box, index);

               hypre_ForSBoxI(j, compute_sbox_a)
                  {
                     compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

                     start  = hypre_SBoxIMin(compute_sbox);

                     hypre_GetSBoxSize(compute_sbox, loop_size);
                     hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                                    A_data_box,  start,  stride,  Ai,
                                    x_data_box,  start,  stride,  xi,
                                    {
                                       xp[xi] -= (Awp[Ai]*xwp[xi] +
                                                  Aep[Ai]*xep[xi]  ) / Ap[Ai];
                                    });
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
 * hypre_CyclicReductionFinalize
 *--------------------------------------------------------------------------*/

int
hypre_CyclicReductionFinalize( void *cyc_red_vdata )
{
   hypre_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   int l;
   int ierr = 0;

   if (cyc_red_data)
   {
      hypre_FreeSBoxArray(cyc_red_data -> base_points);
      for (l = 0; l < ((cyc_red_data -> num_levels) - 1); l++)
      {
         hypre_FreeStructGrid(cyc_red_data -> grid_l[l+1]);
         hypre_FreeSBoxArray(cyc_red_data -> fine_points_l[l]);
         hypre_FreeSBoxArray(cyc_red_data -> coarse_points_l[l]);
         hypre_FreeStructMatrixShell(cyc_red_data -> A_l[l+1]);
         hypre_FreeStructVectorShell(cyc_red_data -> x_l[l+1]);
         hypre_FreeComputePkg(cyc_red_data -> down_compute_pkg_l[l]);
         hypre_FreeComputePkg(cyc_red_data -> up_compute_pkg_l[l]);
      }
      hypre_FreeSBoxArray(cyc_red_data -> fine_points_l[l]);
      hypre_TFree(cyc_red_data -> data);
      hypre_TFree(cyc_red_data -> grid_l);
      hypre_TFree(cyc_red_data -> fine_points_l);
      hypre_TFree(cyc_red_data -> coarse_points_l);
      hypre_TFree(cyc_red_data -> A_l);
      hypre_TFree(cyc_red_data -> x_l);
      hypre_TFree(cyc_red_data -> down_compute_pkg_l);
      hypre_TFree(cyc_red_data -> up_compute_pkg_l);

      hypre_FinalizeTiming(cyc_red_data -> time_index);
      hypre_TFree(cyc_red_data);
   }

   return ierr;
}


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

#define zzz_CycRedMapFineToCoarse(index1, index2, cindex, cstride) \
{\
   zzz_IndexX(index2) =\
      (zzz_IndexX(index1) - zzz_IndexX(cindex)) / zzz_IndexX(cstride);\
   zzz_IndexY(index2) =\
      (zzz_IndexY(index1) - zzz_IndexY(cindex)) / zzz_IndexY(cstride);\
   zzz_IndexZ(index2) =\
      (zzz_IndexZ(index1) - zzz_IndexZ(cindex)) / zzz_IndexZ(cstride);\
}
 
#define zzz_CycRedMapCoarseToFine(index1, index2, cindex, cstride) \
{\
   zzz_IndexX(index2) =\
      zzz_IndexX(index1) * zzz_IndexX(cstride) + zzz_IndexX(cindex);\
   zzz_IndexY(index2) =\
      zzz_IndexY(index1) * zzz_IndexY(cstride) + zzz_IndexY(cindex);\
   zzz_IndexZ(index2) =\
      zzz_IndexZ(index1) * zzz_IndexZ(cstride) + zzz_IndexZ(cindex);\
}

#define zzz_CycRedSetCIndex(base_index, base_stride, level, cdir, cindex) \
{\
   if (level > 0)\
      zzz_SetIndex(cindex, 0, 0, 0);\
   else\
      zzz_CopyIndex(base_index,  cindex);\
   zzz_IndexD(cindex, cdir) = 0;\
}

#define zzz_CycRedSetFIndex(base_index, base_stride, level, cdir, findex) \
{\
   if (level > 0)\
      zzz_SetIndex(findex, 0, 0, 0);\
   else\
      zzz_CopyIndex(base_index,  findex);\
   zzz_IndexD(findex, cdir) = 1;\
}

#define zzz_CycRedSetStride(base_index, base_stride, level, cdir, stride) \
{\
   if (level > 0)\
      zzz_SetIndex(stride, 1, 1, 1);\
   else\
      zzz_CopyIndex(base_stride, stride);\
   zzz_IndexD(stride, cdir) = 2;\
}

/*--------------------------------------------------------------------------
 * zzz_CyclicReductionData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm           *comm;

   int                 num_levels;

   int                 cdir;         /* coarsening direction */
   zzz_Index           base_index;
   zzz_Index           base_stride;

   zzz_StructGrid    **grid_l;
                    
   zzz_SBoxArray      *base_points;
   zzz_SBoxArray     **fine_points_l;
   zzz_SBoxArray     **coarse_points_l;

   zzz_StructMatrix  **A_l;
   zzz_StructVector  **x_l;

   zzz_ComputePkg    **down_compute_pkg_l;
   zzz_ComputePkg    **up_compute_pkg_l;

   int                 time_index;
   int                 solve_flops;

} zzz_CyclicReductionData;

/*--------------------------------------------------------------------------
 * zzz_CyclicReductionInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_CyclicReductionInitialize( MPI_Comm *comm )
{
   zzz_CyclicReductionData *cyc_red_data;

   cyc_red_data = zzz_CTAlloc(zzz_CyclicReductionData, 1);

   (cyc_red_data -> comm) = comm;
   (cyc_red_data -> time_index)  = zzz_InitializeTiming("CyclicReduction");

   /* set defaults */
   zzz_SetIndex((cyc_red_data -> base_index), 0, 0, 0);
   zzz_SetIndex((cyc_red_data -> base_stride), 1, 1, 1);

   return (void *) cyc_red_data;
}

/*--------------------------------------------------------------------------
 * zzz_CycRedNewCoarseOp
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_CycRedNewCoarseOp( zzz_StructMatrix *A,
                       zzz_StructGrid   *coarse_grid,
                       int               cdir        )
{
   zzz_StructMatrix    *Ac;

   zzz_Index           *Ac_stencil_shape;
   zzz_StructStencil   *Ac_stencil;
   int                  Ac_stencil_size;
   int                  Ac_stencil_dim;
   int                  Ac_num_ghost[] = {0, 0, 0, 0, 0, 0};

   int                  i;
   int                  stencil_rank;
 
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

   if (!zzz_StructMatrixSymmetric(A))
   {
      Ac_stencil_size = 3;
      Ac_stencil_shape = zzz_CTAlloc(zzz_Index, Ac_stencil_size);
      for (i = -1; i < 2; i++)
      {
         /* Storage for 3 elements (c,w,e) */
         zzz_SetIndex(Ac_stencil_shape[stencil_rank],i,0,0);
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
      Ac_stencil_shape = zzz_CTAlloc(zzz_Index, Ac_stencil_size);
      for (i = -1; i < 1; i++)
      {

         /* Storage for 2 elements in (c,w) */
         zzz_SetIndex(Ac_stencil_shape[stencil_rank],i,0,0);
         stencil_rank++;
      }
   }

   Ac_stencil = zzz_NewStructStencil(Ac_stencil_dim, Ac_stencil_size,
                                     Ac_stencil_shape);

   Ac = zzz_NewStructMatrix(zzz_StructMatrixComm(A),
                            coarse_grid, Ac_stencil);

   /*-----------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------*/

   zzz_StructMatrixSymmetric(Ac) = zzz_StructMatrixSymmetric(A);

   /*-----------------------------------------------
    * Set number of ghost points
    *-----------------------------------------------*/

   Ac_num_ghost[2*cdir]     = 1;
   Ac_num_ghost[2*cdir + 1] = 1;
   zzz_SetStructMatrixNumGhost(Ac, Ac_num_ghost);

   zzz_InitializeStructMatrix(Ac);
 
   return Ac;
}

/*--------------------------------------------------------------------------
 * zzz_CycRedSetupCoarseOp
 *--------------------------------------------------------------------------*/

int
zzz_CycRedSetupCoarseOp( zzz_StructMatrix *A,
                         zzz_StructMatrix *Ac,
                         zzz_Index         cindex,
                         zzz_Index         cstride )

{
   zzz_Index             index_temp;

   zzz_StructGrid       *cgrid;
   zzz_BoxArray         *cgrid_boxes;
   zzz_Box              *cgrid_box;
   zzz_IndexRef          cstart;
   zzz_Index             stridec;
   zzz_Index             fstart;
   zzz_IndexRef          stridef;
   zzz_Index             loop_size;

   int                   i;
   int                   loopi, loopj, loopk;

   zzz_Box              *A_data_box;
   zzz_Box              *Ac_data_box;

   double               *a_cc, *a_cw, *a_ce;
   double               *ac_cc, *ac_cw, *ac_ce;

   int                   iA, iAm1, iAp1;
   int                   iAc;
                       
   int                   xOffsetA; 
                       
   int                   ierr;

   stridef = cstride;
   zzz_SetIndex(stridec, 1, 1, 1);

   cgrid = zzz_StructMatrixGrid(Ac);
   cgrid_boxes = zzz_StructGridBoxes(cgrid);

   zzz_ForBoxI(i, cgrid_boxes)
   {
      cgrid_box = zzz_BoxArrayBox(cgrid_boxes, i);

      cstart = zzz_BoxIMin(cgrid_box);
      zzz_CycRedMapCoarseToFine(cstart, fstart, cindex, cstride) ;

      A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
      Ac_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(Ac), i);

      /*-----------------------------------------------
       * Extract pointers for 3-point fine grid operator:
       * 
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient
       * a_ce is pointer for east coefficient
       *-----------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      a_cc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      a_cw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,1,0,0);
      a_ce = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      /*-----------------------------------------------
       * Extract pointers for coarse grid operator - always 3-point:
       *
       * If A is symmetric so is Ac.  We build only the
       * lower triangular part (plus diagonal).
       * 
       * ac_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      ac_cc = zzz_StructMatrixExtractPointerByIndex(Ac, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      ac_cw = zzz_StructMatrixExtractPointerByIndex(Ac, i, index_temp);

      if(!zzz_StructMatrixSymmetric(A))
      {
         zzz_SetIndex(index_temp,0,1,0);
         ac_ce = zzz_StructMatrixExtractPointerByIndex(Ac, i, index_temp);
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

      zzz_SetIndex(index_temp,1,0,0);
      xOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 

      /*-----------------------------------------------
       * non-symmetric case
       *-----------------------------------------------*/

      if(!zzz_StructMatrixSymmetric(A))
      {
         zzz_GetBoxSize(cgrid_box, loop_size);
         zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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
         zzz_GetBoxSize(cgrid_box, loop_size);
         zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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

   zzz_AssembleStructMatrix(Ac);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_CyclicReductionSetup
 *--------------------------------------------------------------------------*/

int
zzz_CyclicReductionSetup( void             *cyc_red_vdata,
                          zzz_StructMatrix *A,
                          zzz_StructVector *b,
                          zzz_StructVector *x             )
{
   zzz_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   MPI_Comm             *comm        = (cyc_red_data -> comm);
   int                   cdir        = (cyc_red_data -> cdir);
   zzz_IndexRef          base_index  = (cyc_red_data -> base_index);
   zzz_IndexRef          base_stride = (cyc_red_data -> base_stride);

   int                   num_levels;
   zzz_StructGrid      **grid_l;
   zzz_SBoxArray        *base_points;
   zzz_SBoxArray       **fine_points_l;
   zzz_SBoxArray       **coarse_points_l;
   zzz_StructMatrix    **A_l;
   zzz_StructVector    **x_l;
   zzz_ComputePkg      **down_compute_pkg_l;
   zzz_ComputePkg      **up_compute_pkg_l;

   zzz_BoxArray         *coarsest_boxes;

   zzz_Index             cindex;
   zzz_Index             findex;
   zzz_Index             stride;

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

   zzz_BoxArray         *all_boxes;
   zzz_SBoxArray        *coarse_points;
   int                  *processes;
 
   zzz_SBox             *sbox;
   zzz_Box              *box;
                    
   int                   idmin, idmax;
   int                   i, l;
   int                   flop_divisor;

   int                   x_num_ghost[] = {0, 0, 0, 0, 0, 0};

   int                   ierr;

   /*-----------------------------------------------------
    * Compute a preliminary num_levels value based on the grid
    *-----------------------------------------------------*/

   cdir = zzz_StructStencilDim(zzz_StructMatrixStencil(A)) - 1;

   all_boxes = zzz_StructGridAllBoxes(zzz_StructMatrixGrid(A));
   idmin = zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, 0), cdir);
   idmax = zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, 0), cdir);
   zzz_ForBoxI(i, all_boxes)
   {
      idmin = min(idmin, zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, i), cdir));
      idmax = max(idmax, zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, i), cdir));
   }
   num_levels = zzz_Log2(idmax - idmin + 1) + 2;

   (cyc_red_data -> cdir) = cdir;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid_l    = zzz_TAlloc(zzz_StructGrid *, num_levels);
   grid_l[0] = zzz_StructMatrixGrid(A);

   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      idmin = zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, 0), cdir);
      idmax = zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, 0), cdir);
      zzz_ForBoxI(i, all_boxes)
      {
         idmin = min(idmin, zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, i), cdir));
         idmax = max(idmax, zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, i), cdir));
      }
      if ( idmin == idmax )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen the grid */
      coarse_points = zzz_ProjectBoxArray(zzz_StructGridAllBoxes(grid_l[l]),
                                          cindex, stride);
      all_boxes = zzz_NewBoxArray();
      processes = zzz_TAlloc(int, zzz_SBoxArraySize(coarse_points));
      zzz_ForSBoxI(i, coarse_points)
      {
         sbox = zzz_SBoxArraySBox(coarse_points, i);
         box = zzz_DuplicateBox(zzz_SBoxBox(sbox));
         zzz_CycRedMapFineToCoarse(zzz_BoxIMin(box), zzz_BoxIMin(box),
                                   cindex, stride);
         zzz_CycRedMapFineToCoarse(zzz_BoxIMax(box), zzz_BoxIMax(box),
                                   cindex, stride);
         zzz_AppendBox(box, all_boxes);
         processes[i] = zzz_StructGridProcess(grid_l[l], i);
      }
      grid_l[l+1] =
         zzz_NewAssembledStructGrid(comm, zzz_StructGridDim(grid_l[l]),
                                    all_boxes, processes);
      zzz_FreeSBoxArray(coarse_points);
   }
   num_levels = l + 1;

   (cyc_red_data -> num_levels)      = num_levels;
   (cyc_red_data -> grid_l)          = grid_l;

   /*-----------------------------------------------------
    * Set up base points
    *-----------------------------------------------------*/

   base_points = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid_l[0]),
                                     base_index, base_stride);

   (cyc_red_data -> base_points) = base_points;

   /*-----------------------------------------------------
    * Set up fine and coarse points
    *-----------------------------------------------------*/

   fine_points_l   = zzz_TAlloc(zzz_SBoxArray *,  num_levels);
   coarse_points_l = zzz_TAlloc(zzz_SBoxArray *,  num_levels - 1);

   for (l = 0; l < (num_levels - 1); l++)
   {
      zzz_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      fine_points_l[l]   = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid_l[l]),
                                               findex, stride);
      coarse_points_l[l] = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid_l[l]),
                                               cindex, stride);
   }
   coarsest_boxes = zzz_DuplicateBoxArray(zzz_StructGridBoxes(grid_l[l]));
   fine_points_l[l] = zzz_ConvertToSBoxArray(coarsest_boxes);

   (cyc_red_data -> fine_points_l)   = fine_points_l;
   (cyc_red_data -> coarse_points_l) = coarse_points_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels);
   x_l  = zzz_TAlloc(zzz_StructVector *, num_levels);

   A_l[0] = A;
   x_l[0] = x;

   x_num_ghost[2*cdir]     = 1;
   x_num_ghost[2*cdir + 1] = 1;

   for (l = 0; l < (num_levels - 1); l++)
   {
      A_l[l+1] = zzz_CycRedNewCoarseOp(A_l[l], grid_l[l+1], cdir);

      x_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      zzz_InitializeStructVector(x_l[l+1]);
      zzz_AssembleStructVector(x_l[l+1]);
   }

   (cyc_red_data -> A_l)  = A_l;
   (cyc_red_data -> x_l)  = x_l;

   /*-----------------------------------------------------
    * Set up coarse grid operators
    *-----------------------------------------------------*/

   for (l = 0; l < (num_levels - 1); l++)
   {
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      zzz_CycRedSetupCoarseOp(A_l[l], A_l[l+1], cindex, stride);
   }

   /*----------------------------------------------------------
    * Set up compute packages
    *----------------------------------------------------------*/

   down_compute_pkg_l = zzz_TAlloc(zzz_ComputePkg *, (num_levels - 1));
   up_compute_pkg_l   = zzz_TAlloc(zzz_ComputePkg *, (num_levels - 1));

   for (l = 0; l < (num_levels - 1); l++)
   {
      zzz_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                         &send_box_ranks, &recv_box_ranks,
                         &indt_boxes, &dept_boxes,
                         grid_l[l], zzz_StructMatrixStencil(A_l[l]));
 
      /* down-cycle */
      send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, findex, stride);
      recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, findex, stride);
      indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, cindex, stride);
      dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, cindex, stride);
      down_compute_pkg_l[l] =
         zzz_NewComputePkg(send_sboxes, recv_sboxes,
                           send_box_ranks, recv_box_ranks,
                           indt_sboxes, dept_sboxes,
                           grid_l[l], zzz_StructVectorDataSpace(x_l[l]), 1);

      zzz_FreeBoxArrayArray(send_boxes);
      zzz_FreeBoxArrayArray(recv_boxes);
      zzz_FreeBoxArrayArray(indt_boxes);
      zzz_FreeBoxArrayArray(dept_boxes);

      zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                         &send_box_ranks, &recv_box_ranks,
                         &indt_boxes, &dept_boxes,
                         grid_l[l], zzz_StructMatrixStencil(A_l[l]));

      /* up-cycle */
      send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, cindex, stride);
      recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, cindex, stride);
      indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, findex, stride);
      dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, findex, stride);
      up_compute_pkg_l[l] =
         zzz_NewComputePkg(send_sboxes, recv_sboxes,
                           send_box_ranks, recv_box_ranks,
                           indt_sboxes, dept_sboxes,
                           grid_l[l], zzz_StructVectorDataSpace(x_l[l]), 1);

      zzz_FreeBoxArrayArray(send_boxes);
      zzz_FreeBoxArrayArray(recv_boxes);
      zzz_FreeBoxArrayArray(indt_boxes);
      zzz_FreeBoxArrayArray(dept_boxes);
   }

   (cyc_red_data -> down_compute_pkg_l) = down_compute_pkg_l;
   (cyc_red_data -> up_compute_pkg_l)   = up_compute_pkg_l;

   /*-----------------------------------------------------
    * Compute solve flops
    *-----------------------------------------------------*/

   flop_divisor = (zzz_IndexX(base_stride) *
                   zzz_IndexY(base_stride) *
                   zzz_IndexZ(base_stride)  );
   (cyc_red_data -> solve_flops) =
      zzz_StructVectorGlobalSize(x_l[0])/2/flop_divisor;
   (cyc_red_data -> solve_flops) +=
      5*zzz_StructVectorGlobalSize(x_l[0])/2/flop_divisor;
   for (l = 1; l < (num_levels - 1); l++)
   {
      (cyc_red_data -> solve_flops) +=
         10*zzz_StructVectorGlobalSize(x_l[l])/2;
   }
   (cyc_red_data -> solve_flops) += zzz_StructVectorGlobalSize(x_l[l])/2;

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
         zzz_PrintStructMatrix(filename, A_l[l], 0);
      }
   }
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_CyclicReduction
 *
 * The solution vectors on each level are also used to store the
 * right-hand-side data.  We can do this because of the red-black
 * nature of the algorithm and the fact that the method is exact,
 * allowing one to assume initial guesses of zero on all grid levels.
 *--------------------------------------------------------------------------*/

int
zzz_CyclicReduction( void             *cyc_red_vdata,
                     zzz_StructMatrix *A,
                     zzz_StructVector *b,
                     zzz_StructVector *x             )
{
   zzz_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   int                 num_levels         = (cyc_red_data -> num_levels);
   int                 cdir               = (cyc_red_data -> cdir);
   zzz_IndexRef        base_index         = (cyc_red_data -> base_index);
   zzz_IndexRef        base_stride        = (cyc_red_data -> base_stride);
   zzz_SBoxArray      *base_points        = (cyc_red_data -> base_points);
   zzz_SBoxArray     **fine_points_l      = (cyc_red_data -> fine_points_l);
   zzz_SBoxArray     **coarse_points_l    = (cyc_red_data -> coarse_points_l);
   zzz_StructMatrix  **A_l                = (cyc_red_data -> A_l);
   zzz_StructVector  **x_l                = (cyc_red_data -> x_l);
   zzz_ComputePkg    **down_compute_pkg_l =
      (cyc_red_data -> down_compute_pkg_l);
   zzz_ComputePkg    **up_compute_pkg_l   =
      (cyc_red_data -> up_compute_pkg_l);
                    
   zzz_CommHandle     *comm_handle;
                     
   zzz_SBoxArrayArray *compute_sbox_aa;
   zzz_SBoxArray      *compute_sbox_a;
   zzz_SBox           *compute_sbox;
                     
   zzz_Box            *A_data_box;
   zzz_Box            *x_data_box;
   zzz_Box            *b_data_box;
   zzz_Box            *xc_data_box;
                     
   double             *Ap, *Awp, *Aep;
   double             *xp, *xwp, *xep;
   double             *bp;
   double             *xcp;
                     
   int                 Ai;
   int                 xi;
   int                 bi;
   int                 xci;
                     
   zzz_Index           cindex;
   zzz_Index           stride;
                       
   zzz_Index           index;
   zzz_Index           loop_size;
   zzz_IndexRef        start;
   zzz_Index           startc;
   zzz_Index           stridec;
                     
   int                 compute_i, i, j, l;
   int                 loopi, loopj, loopk;

   int                 ierr;

   zzz_BeginTiming(cyc_red_data -> time_index);

   /*--------------------------------------------------
    * Initialize some things
    *--------------------------------------------------*/

   zzz_SetIndex(stridec, 1, 1, 1);

   /*--------------------------------------------------
    * Copy b into x
    *--------------------------------------------------*/

   compute_sbox_a = base_points;
   zzz_ForSBoxI(i, compute_sbox_a)
   {
      compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);

      x_data_box =
         zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
      b_data_box =
         zzz_BoxArrayBox(zzz_StructVectorDataSpace(b), i);

      xp = zzz_StructVectorBoxData(x, i);
      bp = zzz_StructVectorBoxData(b, i);

      start  = zzz_SBoxIMin(compute_sbox);

      zzz_GetSBoxSize(compute_sbox, loop_size);
      zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      compute_sbox_a = fine_points_l[l];
      zzz_ForSBoxI(i, compute_sbox_a)
      {
         compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);

         A_data_box =
            zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A_l[l]), i);
         x_data_box =
            zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l]), i);

         zzz_SetIndex(index, 0, 0, 0);
         Ap = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
         xp = zzz_StructVectorBoxData(x_l[l], i);

         start  = zzz_SBoxIMin(compute_sbox);

         zzz_GetSBoxSize(compute_sbox, loop_size);
         zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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
               xp = zzz_StructVectorData(x_l[l]);
               comm_handle =
                  zzz_InitializeIndtComputations(down_compute_pkg_l[l], xp);
               compute_sbox_aa =
                  zzz_ComputePkgIndtSBoxes(down_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               zzz_FinalizeIndtComputations(comm_handle);
               compute_sbox_aa =
                  zzz_ComputePkgDeptSBoxes(down_compute_pkg_l[l]);
            }
            break;
         }

         zzz_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

            A_data_box =
               zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A_l[l]), i);
            x_data_box =
               zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l]), i);
            xc_data_box =
               zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l+1]), i);

            xp  = zzz_StructVectorBoxData(x_l[l], i);
            xcp = zzz_StructVectorBoxData(x_l[l+1], i);

            zzz_SetIndex(index, -1, 0, 0);
            Awp = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xwp = zzz_StructVectorBoxData(x_l[l], i) +
               zzz_BoxOffsetDistance(x_data_box, index);

            zzz_SetIndex(index,  1, 0, 0);
            Aep = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xep = zzz_StructVectorBoxData(x_l[l], i) +
               zzz_BoxOffsetDistance(x_data_box, index);

            zzz_ForSBoxI(j, compute_sbox_a)
            {
               compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

               start  = zzz_SBoxIMin(compute_sbox);
               zzz_CycRedMapFineToCoarse(start, startc, cindex, stride);

               zzz_GetSBoxSize(compute_sbox, loop_size);
               zzz_BoxLoop3(loopi, loopj, loopk, loop_size,
                            A_data_box,  start,  stride,  Ai,
                            x_data_box,  start,  stride,  xi,
                            xc_data_box, startc, stridec, xci,
                            {
                               xcp[xci] =
                                  xp[xi] - Awp[Ai]*xwp[xi] - Aep[Ai]*xep[xi];
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
      zzz_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      zzz_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      compute_sbox_a = coarse_points_l[l];
      zzz_ForSBoxI(i, compute_sbox_a)
      {
         compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);

         x_data_box =
            zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l]), i);
         xc_data_box =
            zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l+1]), i);

         xp  = zzz_StructVectorBoxData(x_l[l], i);
         xcp = zzz_StructVectorBoxData(x_l[l+1], i);

         start  = zzz_SBoxIMin(compute_sbox);
         zzz_CycRedMapFineToCoarse(start, startc, cindex, stride);

         zzz_GetSBoxSize(compute_sbox, loop_size);
         zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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
               xp = zzz_StructVectorData(x_l[l]);
               comm_handle =
                  zzz_InitializeIndtComputations(up_compute_pkg_l[l], xp);
               compute_sbox_aa =
                  zzz_ComputePkgIndtSBoxes(up_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               zzz_FinalizeIndtComputations(comm_handle);
               compute_sbox_aa =
                  zzz_ComputePkgDeptSBoxes(up_compute_pkg_l[l]);
            }
            break;
         }

         zzz_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

            A_data_box =
               zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A_l[l]), i);
            x_data_box =
               zzz_BoxArrayBox(zzz_StructVectorDataSpace(x_l[l]), i);

            zzz_SetIndex(index, 0, 0, 0);
            Ap = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xp = zzz_StructVectorBoxData(x_l[l], i);

            zzz_SetIndex(index, -1, 0, 0);
            Awp = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xwp = zzz_StructVectorBoxData(x_l[l], i) +
               zzz_BoxOffsetDistance(x_data_box, index);

            zzz_SetIndex(index,  1, 0, 0);
            Aep = zzz_StructMatrixExtractPointerByIndex(A_l[l], i, index);
            xep = zzz_StructVectorBoxData(x_l[l], i) +
               zzz_BoxOffsetDistance(x_data_box, index);

            zzz_ForSBoxI(j, compute_sbox_a)
            {
               compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

               start  = zzz_SBoxIMin(compute_sbox);

               zzz_GetSBoxSize(compute_sbox, loop_size);
               zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
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

   zzz_IncFLOPCount(cyc_red_data -> solve_flops);
   zzz_EndTiming(cyc_red_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_CyclicReductionSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_CyclicReductionSetBase( void      *cyc_red_vdata,
                            zzz_Index  base_index,
                            zzz_Index  base_stride )
{
   zzz_CyclicReductionData *cyc_red_data = cyc_red_vdata;
   int                      d;
   int                      ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((cyc_red_data -> base_index),  d) =
         zzz_IndexD(base_index,  d);
      zzz_IndexD((cyc_red_data -> base_stride), d) =
         zzz_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_CyclicReductionFinalize
 *--------------------------------------------------------------------------*/

int
zzz_CyclicReductionFinalize( void *cyc_red_vdata )
{
   zzz_CyclicReductionData *cyc_red_data = cyc_red_vdata;

   int l;
   int ierr;

   if (cyc_red_data)
   {
      zzz_FreeSBoxArray(cyc_red_data -> base_points);
      for (l = 0; l < ((cyc_red_data -> num_levels) - 1); l++)
      {
         zzz_FreeStructGrid(cyc_red_data -> grid_l[l+1]);
         zzz_FreeSBoxArray(cyc_red_data -> fine_points_l[l]);
         zzz_FreeSBoxArray(cyc_red_data -> coarse_points_l[l]);
         zzz_FreeStructMatrix(cyc_red_data -> A_l[l+1]);
         zzz_FreeStructVector(cyc_red_data -> x_l[l+1]);
         zzz_FreeComputePkg(cyc_red_data -> down_compute_pkg_l[l]);
         zzz_FreeComputePkg(cyc_red_data -> up_compute_pkg_l[l]);
      }
      zzz_FreeSBoxArray(cyc_red_data -> fine_points_l[l]);
      zzz_TFree(cyc_red_data -> grid_l);
      zzz_TFree(cyc_red_data -> fine_points_l);
      zzz_TFree(cyc_red_data -> coarse_points_l);
      zzz_TFree(cyc_red_data -> A_l);
      zzz_TFree(cyc_red_data -> x_l);
      zzz_TFree(cyc_red_data -> down_compute_pkg_l);
      zzz_TFree(cyc_red_data -> up_compute_pkg_l);

      zzz_FinalizeTiming(cyc_red_data -> time_index);
      zzz_TFree(cyc_red_data);
   }

   return ierr;
}


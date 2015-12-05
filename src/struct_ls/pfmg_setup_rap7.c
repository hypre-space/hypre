/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.23 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * Macro to "change coordinates".  This routine is written as though
 * coarsening is being done in the z-direction.  This macro is used to
 * allow for coarsening to be done in the x- and y-directions also.
 *--------------------------------------------------------------------------*/

#define MapIndex(in_index, cdir, out_index)                     \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 2);   \
   cdir = (cdir + 1) % 3;                                       \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 0);   \
   cdir = (cdir + 1) % 3;                                       \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 1);   \
   cdir = (cdir + 1) % 3;

/*--------------------------------------------------------------------------
 * hypre_PFMGCreateCoarseOp7 
 *    Sets up new coarse grid operator stucture. Fine grid
 *    operator is 7pt and so is coarse, i.e. non-Galerkin.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_PFMGCreateCoarseOp7( hypre_StructMatrix *R,
                           hypre_StructMatrix *A,
                           hypre_StructMatrix *P,
                           hypre_StructGrid   *coarse_grid,
                           HYPRE_Int           cdir        )
{
   hypre_StructMatrix    *RAP;

   hypre_Index           *RAP_stencil_shape;
   hypre_StructStencil   *RAP_stencil;
   HYPRE_Int              RAP_stencil_size;
   HYPRE_Int              RAP_stencil_dim;
   HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   hypre_Index            index_temp;
   HYPRE_Int              k, j, i;
   HYPRE_Int              stencil_rank;
 
   RAP_stencil_dim = 3;

   /*-----------------------------------------------------------------------
    * Define RAP_stencil
    *-----------------------------------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------------------------------
    * non-symmetric case
    *-----------------------------------------------------------------------*/

   if (!hypre_StructMatrixSymmetric(A))
   {

      /*--------------------------------------------------------------------
       * 7 point coarse grid stencil 
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 7;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (k = -1; k < 2; k++)
      {
         for (j = -1; j < 2; j++)
         {
            for (i = -1; i < 2; i++)
            {

               /*--------------------------------------------------------------
                * Storage for 7 elements (c,w,e,n,s,a,b)
                *--------------------------------------------------------------*/
               if (i*j == 0 && i*k == 0 && j*k == 0)
               {
                  hypre_SetIndex(index_temp,i,j,k);
                  MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
                  stencil_rank++;
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * symmetric case
    *-----------------------------------------------------------------------*/

   else
   {

      /*--------------------------------------------------------------------
       * 7 point coarse grid stencil
       * Only store the lower triangular part + diagonal = 4 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicographic ordering.
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 4;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (k = -1; k < 1; k++)
      {
         for (j = -1; j < 1; j++)
         {
            for (i = -1; i < 1; i++)
            {

               /*--------------------------------------------------------------
                * Store 4 elements in (c,w,s,b)
                *--------------------------------------------------------------*/
               if (i*j == 0 && i*k == 0 && j*k == 0)
               {
                  hypre_SetIndex(index_temp,i,j,k);
                  MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
                  stencil_rank++;
               }
            }
         }
      }
   }

   RAP_stencil = hypre_StructStencilCreate(RAP_stencil_dim, RAP_stencil_size,
                                           RAP_stencil_shape);

   RAP = hypre_StructMatrixCreate(hypre_StructMatrixComm(A),
                                  coarse_grid, RAP_stencil);

   hypre_StructStencilDestroy(RAP_stencil);

   /*-----------------------------------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------------------------------*/
   hypre_StructMatrixSymmetric(RAP) = hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------------------------------
    * Set number of ghost points - one one each boundary
    *-----------------------------------------------------------------------*/
   hypre_StructMatrixSetNumGhost(RAP, RAP_num_ghost);

   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGBuildCoarseOp7
 *    Sets up new coarse grid operator stucture. Fine grid operator is 7pt and
 *    so is coarse, i.e. non-Galerkin.
 *
 *    Uses the non-Galerkin strategy from Ashby & Falgout's original ParFlow
 *    algorithm.  For constant_coefficient==2, see [issue663].
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGBuildCoarseOp7( hypre_StructMatrix *A,
                          hypre_StructMatrix *P,
                          hypre_StructMatrix *R,
                          HYPRE_Int           cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride,
                          hypre_StructMatrix *RAP     )
{
   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *fgrid;
   hypre_BoxArray       *fgrid_boxes;
   hypre_Box            *fgrid_box;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   HYPRE_Int            *cgrid_ids;
   hypre_IndexRef        cstart, bfstart, stridef;
   hypre_Index           fstart, bcstart, stridec;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient;

   HYPRE_Int             fi, ci, fbi;
   HYPRE_Int             loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *RAP_dbox;

   hypre_BoxArray       *bdy_boxes, *tmp_boxes;
   hypre_Box            *bdy_box, *fcbox;

   double               *pb, *pa;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_cb, *a_ca;

   double               *rap_cc, *rap_cw, *rap_ce, *rap_cs, *rap_cn;
   double               *rap_cb, *rap_ca;
   double                west, east, south, north;
   double                center_int, center_bdy;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iPm1, iPp1;
                      
   HYPRE_Int             OffsetA; 
   HYPRE_Int             OffsetP; 
                      
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_boxes = hypre_StructGridBoxes(fgrid);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   hypre_assert( hypre_StructMatrixConstantCoefficient(A) == constant_coefficient );
   if ( constant_coefficient==0 )
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) == 0 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) == 0 );
   }
   else /* 1 or 2 */
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) == 1 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) == 1 );
   }

   fcbox = hypre_BoxCreate();
   bdy_boxes = hypre_BoxArrayCreate(0);
   tmp_boxes = hypre_BoxArrayCreate(0);

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
      fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);

      cstart = hypre_BoxIMin(cgrid_box);
      hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
      P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
      RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pb is pointer for weight for f-point below c-point 
       * pa is pointer for weight for f-point above c-point 
       *-----------------------------------------------------------------*/

      hypre_SetIndex(index_temp,0,0,-1);
      MapIndex(index_temp, cdir, index);
      pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

      hypre_SetIndex(index_temp,0,0,1);
      MapIndex(index_temp, cdir, index);
      pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
         hypre_BoxOffsetDistance(P_dbox, index);
 
      /*-----------------------------------------------------------------
       * Extract pointers for 7-point fine grid operator:
       * 
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient
       * a_ce is pointer for east coefficient
       * a_cs is pointer for south coefficient
       * a_cn is pointer for north coefficient
       * a_cb is pointer for below coefficient
       * a_ca is pointer for above coefficient
       *-----------------------------------------------------------------*/

      hypre_SetIndex(index_temp,0,0,0);
      MapIndex(index_temp, cdir, index);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,-1,0,0);
      MapIndex(index_temp, cdir, index);
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,1,0,0);
      MapIndex(index_temp, cdir, index);
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,0,-1,0);
      MapIndex(index_temp, cdir, index);
      a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,0,1,0);
      MapIndex(index_temp, cdir, index);
      a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,0,0,-1);
      MapIndex(index_temp, cdir, index);
      a_cb = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex(index_temp,0,0,1);
      MapIndex(index_temp, cdir, index);
      a_ca = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      /*-----------------------------------------------------------------
       * Extract pointers for coarse grid operator
       * rap_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------------------------*/

      hypre_SetIndex(index_temp,0,0,0);
      MapIndex(index_temp, cdir, index);
      rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,-1,0,0);
      MapIndex(index_temp, cdir, index);
      rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,1,0,0);
      MapIndex(index_temp, cdir, index);
      rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,0,-1,0);
      MapIndex(index_temp, cdir, index);
      rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,0,1,0);
      MapIndex(index_temp, cdir, index);
      rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,0,0,-1);
      MapIndex(index_temp, cdir, index);
      rap_cb = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex(index_temp,0,0,1);
      MapIndex(index_temp, cdir, index);
      rap_ca = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets are used in refering to data associated with other points. 
       *-----------------------------------------------------------------*/

      hypre_SetIndex(index_temp,0,0,1);
      MapIndex(index_temp, cdir, index);

      OffsetP = hypre_BoxOffsetDistance(P_dbox,index);
      OffsetA = hypre_BoxOffsetDistance(A_dbox,index);

      /*--------------------------------------------------------------
       * Loop for symmetric 7-point fine grid operator; produces a
       * symmetric 7-point coarse grid operator. 
       *--------------------------------------------------------------*/

      if ( constant_coefficient==0 )
      {
         hypre_BoxGetSize(cgrid_box, loop_size);

         hypre_BoxLoop3Begin(loop_size,
                             P_dbox, cstart, stridec, iP,
                             A_dbox, fstart, stridef, iA,
                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iA,iAc,iAm1,iAp1,iPm1,iPp1,west,east,south,north
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop3For(loopi, loopj, loopk, iP, iA, iAc)
         {
            iAm1 = iA - OffsetA;
            iAp1 = iA + OffsetA;

            iPm1 = iP - OffsetP;
            iPp1 = iP + OffsetP;

            rap_cb[iAc] = a_cb[iA] * pa[iPm1];
            rap_ca[iAc] = a_ca[iA] * pb[iPp1];

            west  = a_cw[iA] + 0.5 * a_cw[iAm1] + 0.5 * a_cw[iAp1];
            east  = a_ce[iA] + 0.5 * a_ce[iAm1] + 0.5 * a_ce[iAp1];
            south = a_cs[iA] + 0.5 * a_cs[iAm1] + 0.5 * a_cs[iAp1];
            north = a_cn[iA] + 0.5 * a_cn[iAm1] + 0.5 * a_cn[iAp1];

            /*-----------------------------------------------------
             * Prevent non-zero entries reaching off grid
             *-----------------------------------------------------*/
            if(a_cw[iA] == 0.0) west = 0.0;
            if(a_ce[iA] == 0.0) east = 0.0;
            if(a_cs[iA] == 0.0) south = 0.0;
            if(a_cn[iA] == 0.0) north = 0.0;

            rap_cw[iAc] = west;
            rap_ce[iAc] = east;
            rap_cs[iAc] = south;
            rap_cn[iAc] = north;

            rap_cc[iAc] = a_cc[iA] 
               + a_cw[iA] + a_ce[iA] + a_cs[iA] + a_cn[iA]
               + a_cb[iA] * pb[iP] + a_ca[iA] * pa[iP]
               - west - east - south - north;
         }
         hypre_BoxLoop3End(iP, iA, iAc);
      }

      else if ( constant_coefficient==1 )
      {
         rap_cb[0] = rap_ca[0] = a_cb[0] * pa[0];

         rap_cw[0] = rap_ce[0] = 2.0*a_cw[0];
         rap_cs[0] = rap_cn[0] = 2.0*a_cs[0];

         rap_cc[0] = a_cc[0] - 2.0*( a_cw[0] + a_cs[0] - rap_cb[0] );
      }

      else if ( constant_coefficient==2 )
      {
         /* NOTE: This does not reduce to either of the above operators unless
          * the row sum is zero and the interpolation weights are 1/2 */

         rap_cb[0] = rap_ca[0] = 0.5*a_cb[0];

         rap_cw[0] = rap_ce[0] = 2.0*a_cw[0];
         rap_cs[0] = rap_cn[0] = 2.0*a_cs[0];

         center_int = 3.0*a_cb[0];
         center_bdy = 0.5*a_cb[0] + (a_cw[0] + a_cs[0] + a_cb[0]);

         hypre_BoxGetSize(cgrid_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             A_dbox, fstart, stridef, iA,
                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA,iAc
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, iA, iAc)
         {
            rap_cc[iAc] = 2.0*a_cc[iA] + center_int;
         }
         hypre_BoxLoop2End(iA, iAc);

         hypre_CopyBox(cgrid_box, fcbox);
         hypre_StructMapCoarseToFine(hypre_BoxIMin(fcbox), cindex, cstride,
                                     hypre_BoxIMin(fcbox));
         hypre_StructMapCoarseToFine(hypre_BoxIMax(fcbox), cindex, cstride,
                                     hypre_BoxIMax(fcbox));
         hypre_BoxArraySetSize(bdy_boxes, 0);
         if (hypre_BoxIMinD(fcbox, cdir) == hypre_BoxIMinD(fgrid_box, cdir))
         {
            hypre_BoxBoundaryIntersect(fcbox, fgrid, cdir, -1, bdy_boxes);
         }
         if (hypre_BoxIMaxD(fcbox, cdir) == hypre_BoxIMaxD(fgrid_box, cdir))
         {
            hypre_BoxBoundaryIntersect(fcbox, fgrid, cdir, 1, tmp_boxes);
            hypre_AppendBoxArray(tmp_boxes, bdy_boxes);
         }

         hypre_ForBoxI(fbi, bdy_boxes)
         {
            bdy_box = hypre_BoxArrayBox(bdy_boxes, fbi);

            hypre_BoxGetSize(bdy_box, loop_size);
            bfstart = hypre_BoxIMin(bdy_box);
            hypre_StructMapFineToCoarse(bfstart, cindex, cstride, bcstart);
            hypre_BoxLoop2Begin(loop_size,
                                A_dbox, bfstart, stridef, iA,
                                RAP_dbox, bcstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA,iAc
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, iA, iAc)
            {
               rap_cc[iAc] -= 0.5*a_cc[iA] + center_bdy;
            }
            hypre_BoxLoop2End(iA, iAc);
         }
      }

   } /* end ForBoxI */

   hypre_BoxDestroy(fcbox);
   hypre_BoxArrayDestroy(bdy_boxes);
   hypre_BoxArrayDestroy(tmp_boxes);

   return hypre_error_flag;
}

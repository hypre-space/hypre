/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * Macro to "change coordinates".  This routine is written as though
 * coarsening is being done in the y-direction.  This macro is used to
 * allow for coarsening to be done in the x-direction also.
 *--------------------------------------------------------------------------*/

#define MapIndex(in_index, cdir, out_index)                     \
   hypre_IndexD(out_index, 2)    = hypre_IndexD(in_index, 2);   \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 1);   \
   cdir = (cdir + 1) % 2;                                       \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 0);   \
   cdir = (cdir + 1) % 2;

/*--------------------------------------------------------------------------
 * hypre_PFMGCreateCoarseOp5 
 *    Sets up new coarse grid operator stucture. Fine grid
 *    operator is 5pt and so is coarse, i.e. non-Galerkin.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_PFMGCreateCoarseOp5( hypre_StructMatrix *R,
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
   HYPRE_Int              j, i;
   HYPRE_Int              stencil_rank;
 
   RAP_stencil_dim = 2;

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
       * 5 point coarse grid stencil 
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 5;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (j = -1; j < 2; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Storage for 5 elements (c,w,e,n,s)
             *--------------------------------------------------------------*/
            if (i*j == 0)
            {
               hypre_SetIndex(index_temp,i,j,0);
               MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
               stencil_rank++;
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
       * 5 point coarse grid stencil
       * Only store the lower triangular part + diagonal = 3 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicographic ordering.
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 3;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (j = -1; j < 1; j++)
      {
         for (i = -1; i < 1; i++)
         {

            /*--------------------------------------------------------------
             * Store 3 elements in (c,w,s)
             *--------------------------------------------------------------*/
            if( i*j == 0 )
            {
               hypre_SetIndex(index_temp,i,j,0);
               MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
               stencil_rank++;
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
 * hypre_PFMGBuildCoarseOp5
 *    Sets up new coarse grid operator stucture. Fine grid
 *    operator is 5pt and so is coarse, i.e. non-Galerkin.
 *
 *    Uses the non-Galerkin strategy from Ashby & Falgout's
 *    original ParFlow algorithm (See LLNL Tech. Rep. UCRL-JC-122359).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGBuildCoarseOp5( hypre_StructMatrix *A,
                          hypre_StructMatrix *P,
                          hypre_StructMatrix *R,
                          HYPRE_Int           cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride,
                          hypre_StructMatrix *RAP     )
{
   hypre_StructGrid     *fgrid;
   hypre_BoxArray       *fgrid_boxes;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;

   HYPRE_Int             constant_coefficient;
   HYPRE_Int             fi, ci;

   HYPRE_Int             ierr = 0;

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_boxes = hypre_StructGridBoxes(fgrid);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   hypre_assert( constant_coefficient==0 || constant_coefficient==1 || constant_coefficient==2 );
   hypre_assert( hypre_StructMatrixConstantCoefficient(A) == constant_coefficient );
   if ( constant_coefficient==0 )
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) == 0 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) == 0 );
   }
   else if (constant_coefficient==2 )
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) == 1 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) == 1 );
   }
   else
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) == 1 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) == 1 );
   }
      
   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      /*--------------------------------------------------------------
       * Loop for symmetric 5-point fine grid operator; produces a
       * symmetric 5-point coarse grid operator. 
       *--------------------------------------------------------------*/

      switch (constant_coefficient)
      {
         case 0:
            ierr += hypre_PFMGBuildCoarseOp5_onebox_CC0(
               fi, ci, A, P, R, cdir, cindex, cstride, RAP );
            break;

         case 1:
            ierr += hypre_PFMGBuildCoarseOp5_onebox_CC1(
               fi, ci, A, P, R, cdir, cindex, cstride, RAP );
            break;
         default:  /* 2 */
            ierr += hypre_PFMGBuildCoarseOp5_onebox_CC2(
               fi, ci, A, P, R, cdir, cindex, cstride, RAP );
            break;
      }

   } /* end ForBoxI */

   return ierr;
}

/* inside of the box loop of hypre_PFMGBuildCoarseOp5, for constant_coefficient=0 */
HYPRE_Int
hypre_PFMGBuildCoarseOp5_onebox_CC0(
   HYPRE_Int fi,
   HYPRE_Int ci,
   hypre_StructMatrix *A,
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
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *RAP_dbox;
   double               *pa, *pb;
   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_ce, *rap_cn;
   double                west, east;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iPm1, iPp1;
                      
   HYPRE_Int             yOffsetA; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   /*hypre_printf("2d 0\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_boxes = hypre_StructGridBoxes(fgrid);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

/*
  fi = 0;
  hypre_ForBoxI(ci, cgrid_boxes)
  {
  while (fgrid_ids[fi] != cgrid_ids[ci])
  {
  fi++;
  }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
   fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point 
    * pb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
      hypre_BoxOffsetDistance(P_dbox, index);
 
   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    * 
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
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

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points. 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox,index);
   yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);

   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 5-point coarse grid operator. 
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   hypre_BoxLoop3Begin(loop_size,
                       P_dbox, cstart, stridec, iP,
                       A_dbox, fstart, stridef, iA,
                       RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iA,iAc,iAm1,iAp1,iPm1,iPp1,west,east
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop3For(loopi, loopj, loopk, iP, iA, iAc)
   {
      iAm1 = iA - yOffsetA;
      iAp1 = iA + yOffsetA;

      iPm1 = iP - yOffsetP;
      iPp1 = iP + yOffsetP;

      rap_cs[iAc] = a_cs[iA] * pa[iPm1];
      rap_cn[iAc] = a_cn[iA] * pb[iPp1];

      west = a_cw[iA] + 0.5 * a_cw[iAm1] + 0.5 * a_cw[iAp1];
      east = a_ce[iA] + 0.5 * a_ce[iAm1] + 0.5 * a_ce[iAp1];

      /*-----------------------------------------------------
       * Prevent non-zero entries reaching off grid
       *-----------------------------------------------------*/
      if(a_cw[iA] == 0.0) west = 0.0;
      if(a_ce[iA] == 0.0) east = 0.0;
               
      rap_cw[iAc] = west;
      rap_ce[iAc] = east;

      rap_cc[iAc] = a_cc[iA] + a_cw[iA] + a_ce[iA]
         + a_cs[iA] * pb[iP] + a_cn[iA] * pa[iP]
         - west - east;
   }
   hypre_BoxLoop3End(iP, iA, iAc);

/*      }*/ /* end ForBoxI */

   return ierr;
}

/* inside of the box loop of hypre_PFMGBuildCoarseOp5, for constant_coefficient=1 */
HYPRE_Int
hypre_PFMGBuildCoarseOp5_onebox_CC1(
   HYPRE_Int fi,
   HYPRE_Int ci,
   hypre_StructMatrix *A,
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
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_ce, *rap_cn;
   double                west, east;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iPm1, iPp1;
                      
   HYPRE_Int             yOffsetA; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   /*hypre_printf("2d 1\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_boxes = hypre_StructGridBoxes(fgrid);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

/*
  fi = 0;
  hypre_ForBoxI(ci, cgrid_boxes)
  {
  while (fgrid_ids[fi] != cgrid_ids[ci])
  {
  fi++;
  }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
   fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point 
    * pb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
      hypre_CCBoxOffsetDistance(P_dbox, index);
 
   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    * 
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
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

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points. 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   yOffsetA = hypre_CCBoxOffsetDistance(A_dbox,index); 
   yOffsetP = hypre_CCBoxOffsetDistance(P_dbox,index);

   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 5-point coarse grid operator. 
    *--------------------------------------------------------------*/

   iP = hypre_CCBoxIndexRank(P_dbox,cstart);
   iA = hypre_CCBoxIndexRank(A_dbox,fstart);
   iAc = hypre_CCBoxIndexRank(RAP_dbox, cstart);

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iPm1 = iP - yOffsetP;
   iPp1 = iP + yOffsetP;

   rap_cs[iAc] = a_cs[iA] * pa[iPm1];
   rap_cn[iAc] = a_cn[iA] * pb[iPp1];

   west = a_cw[iA] + 0.5 * a_cw[iAm1] + 0.5 * a_cw[iAp1];
   east = a_ce[iA] + 0.5 * a_ce[iAm1] + 0.5 * a_ce[iAp1];

   /*-----------------------------------------------------
    * Prevent non-zero entries reaching off grid
    *-----------------------------------------------------*/
   if(a_cw[iA] == 0.0) west = 0.0;
   if(a_ce[iA] == 0.0) east = 0.0;
               
   rap_cw[iAc] = west;
   rap_ce[iAc] = east;

   rap_cc[iAc] = a_cc[iA] + a_cw[iA] + a_ce[iA]
      + a_cs[iA] * pb[iP] + a_cn[iA] * pa[iP]
      - west - east;

/*      }*/ /* end ForBoxI */

   return ierr;
}

/* inside of the box loop of hypre_PFMGBuildCoarseOp5, for constant_coefficient=2 */
HYPRE_Int
hypre_PFMGBuildCoarseOp5_onebox_CC2(
   HYPRE_Int fi,
   HYPRE_Int ci,
   hypre_StructMatrix *A,
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
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             loopi, loopj, loopk;
   HYPRE_Int             fbi, floopi, floopj, floopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *RAP_dbox;
   hypre_BoxArray       *fboundarys;
   hypre_BoxArray       *fboundaryn;
   hypre_Box            *fg_bdy_box;
   hypre_Index           box_index;

   double               *pa;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_ce, *rap_cn;
   double                west, east, north, south;
   double                diag, diagcorr, diagm, diagp;

   HYPRE_Int             iA, iAm1, iAp1, iA_offd;
   HYPRE_Int             iAc;
                      
   HYPRE_Int             yOffsetA; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;
   HYPRE_Int             bdy;

   /*hypre_printf("2d 2\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_boxes = hypre_StructGridBoxes(fgrid);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

/*
  fi = 0;
  hypre_ForBoxI(ci, cgrid_boxes)
  {
  while (fgrid_ids[fi] != cgrid_ids[ci])
  {
  fi++;
  }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
   fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
 
   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    * 
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
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

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points. 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox,index);
   yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);

   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 5-point coarse grid operator. 
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);


   /* We're not doing a true RAP computation in this case.
      The new (coarsened) A is designed to keep it
      constant-coefficient in its offdiagonal elements. */

   /* new offdiagonal (constant) elements in the uncoarsened directions ... */
   iA_offd = hypre_CCBoxIndexRank_noargs();  /* really, 0 */
   east = 2*a_ce[iA_offd];
   west = 2*a_cw[iA_offd];

   north = 0.5*a_cn[iA_offd];
   south = 0.5*a_cs[iA_offd];

   rap_cw[iA_offd] = west;
   rap_ce[iA_offd] = east;
   rap_cs[iA_offd] = south;
   rap_cn[iA_offd] = north;
   diag = -west - east -north -south;
   diagcorr = a_cw[iA_offd] + a_ce[iA_offd] + a_cs[iA_offd] + a_cn[iA_offd];

   fboundaryn = hypre_BoxArrayCreate(0);
   fboundarys = hypre_BoxArrayCreate(0);
   hypre_BoxBoundaryDG( fgrid_box, fgrid, fboundarys, fboundaryn, cdir );
   /* ... cgrid_box comes from the grid, so there are no ghost zones
      involved here */
   /* new diagonal (variable) elements...*/
   hypre_BoxLoop2Begin(loop_size,
                       A_dbox, fstart, stridef, iA,
                       RAP_dbox, cstart, stridec, iAc);
#if 0 /* box_index and fg_bdy_box may be a problem here */
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA,iAc,iAm1,iAp1,diagm,diagp,bdy,floopi,floopj,floopk,box_index,fbi,fg_bdy_box
#include "hypre_box_smp_forloop.h"
#endif
   hypre_BoxLoopSetOneBlock(); /* this is needed (because of bdy logic?) */
   hypre_BoxLoop2For(loopi, loopj, loopk, iA, iAc)
   {
      iAm1 = iA - yOffsetA;
      iAp1 = iA + yOffsetA;
      diagm = a_cc[iAm1] + diagcorr;
      diagp = a_cc[iAp1] + diagcorr;
      rap_cc[iAc] = a_cc[iA] + diagcorr + diag + 0.5*( diagm+diagp );

      bdy = 0;  /* so we don't treat this point as a boundary pt twice */
      floopi= fstart[0] + loopi*stridef[0];
      floopj= fstart[1] + loopj*stridef[1];
      floopk= fstart[2] + loopk*stridef[2];
      hypre_SetIndex(box_index, floopi, floopj, floopk);

      hypre_ForBoxI(fbi, fboundarys)
      {
         fg_bdy_box = hypre_BoxArrayBox( fboundarys, fbi);
         if ( hypre_IndexInBoxP( box_index, fg_bdy_box ) && bdy==0 )
         {  /* we're in a boundary (in the south direction) */
            rap_cc[iAc] -= south;
            rap_cc[iAc] -= 0.5*diagm;
            bdy = 1;
            break;
         }
      }
      if ( bdy == 0 )
         hypre_ForBoxI(fbi, fboundaryn)
         {
            fg_bdy_box = hypre_BoxArrayBox( fboundaryn, fbi);
            if ( hypre_IndexInBoxP( box_index, fg_bdy_box ) && bdy==0 )
            {  /* we're in a boundary (in the north direction) */
               rap_cc[iAc] -= north;
               rap_cc[iAc] -= 0.5*diagp;
               bdy = 1;
               break;
            }
         }
   }
   hypre_BoxLoop2End(iA, iAc);

   hypre_BoxArrayDestroy(fboundaryn);
   hypre_BoxArrayDestroy(fboundarys);

/*      }*/ /* end ForBoxI */

   return ierr;
}


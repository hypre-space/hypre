/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.16 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * Macro to "change coordinates".  This routine is written as though
 * coarsening is being done in the y-direction.  This macro is used to
 * allow for coarsening to be done in the x-direction also.
 *--------------------------------------------------------------------------*/

#define MapIndex(in_index, cdir, out_index) \
hypre_IndexD(out_index, 2)    = hypre_IndexD(in_index, 2);\
hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 1);\
cdir = (cdir + 1) % 2;\
hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 0);\
cdir = (cdir + 1) % 2;

/*--------------------------------------------------------------------------
 * hypre_PFMG2CreateRAPOp 
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_PFMG2CreateRAPOp( hypre_StructMatrix *R,
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
       * 5 or 9 point fine grid stencil produces 9 point RAP
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 9;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (j = -1; j < 2; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Storage for 9 elements (c,w,e,n,s,sw,se,nw,se)
             *--------------------------------------------------------------*/
            hypre_SetIndex(index_temp,i,j,0);
            MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
            stencil_rank++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    * symmetric case
    *-----------------------------------------------------------------------*/

   else
   {

      /*--------------------------------------------------------------------
       * 5 or 9 point fine grid stencil produces 9 point RAP
       * Only store the lower triangular part + diagonal = 5 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicographic ordering.
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 5;
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);
      for (j = -1; j < 1; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Store 5 elements in (c,w,s,sw,se)
             *--------------------------------------------------------------*/
            if( i+j <=0 )
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
 * Routines to build RAP. These routines are fairly general
 *  1) No assumptions about symmetry of A
 *  2) No assumption that R = transpose(P)
 *  3) 5 or 9-point fine grid A 
 *
 * I am, however, assuming that the c-to-c interpolation is the identity.
 *
 * I've written two routines - hypre_PFMG2BuildRAPSym to build the
 * lower triangular part of RAP (including the diagonal) and
 * hypre_PFMG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the
 * first routine would be called. With full storage both would need to
 * be called.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMG2BuildRAPSym( hypre_StructMatrix *A,
                        hypre_StructMatrix *P,
                        hypre_StructMatrix *R,
                        HYPRE_Int           cdir,
                        hypre_Index         cindex,
                        hypre_Index         cstride,
                        hypre_StructMatrix *RAP     )
{
   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;

   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;

   HYPRE_Int             constant_coefficient;
   HYPRE_Int             constant_coefficient_A;
   HYPRE_Int             fi, ci;
   HYPRE_Int             ierr = 0;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);
   hypre_assert( constant_coefficient==0 || constant_coefficient==1 );
   hypre_assert( hypre_StructMatrixConstantCoefficient(R) == constant_coefficient );
   hypre_assert( hypre_StructMatrixConstantCoefficient(P) == constant_coefficient );
   if (constant_coefficient==1 )
   {
      hypre_assert( constant_coefficient_A==1 );
   }
   else
   {
      hypre_assert( constant_coefficient_A==0 || constant_coefficient_A==2 );
   }

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         /*-----------------------------------------------------------------
          * Switch statement to direct control to apropriate BoxLoop depending
          * on stencil size. Default is full 9-point.
          *-----------------------------------------------------------------*/

         switch (fine_stencil_size)
         {

            /*--------------------------------------------------------------
             * Loop for symmetric 5-point fine grid operator; produces a
             * symmetric 9-point coarse grid operator. We calculate only the
             * lower triangular stencil entries: (southwest, south, southeast,
             * west, and center).
             *--------------------------------------------------------------*/

            case 5:

            if ( constant_coefficient==1 )
            {
               ierr += hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               ierr += hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

            /*--------------------------------------------------------------
             * Loop for symmetric 9-point fine grid operator; produces a
             * symmetric 9-point coarse grid operator. We calculate only the
             * lower triangular stencil entries: (southwest, south, southeast,
             * west, and center).
             *--------------------------------------------------------------*/

            default:

            if ( constant_coefficient==1 )
            {
               ierr += hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               ierr += hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         } /* end switch statement */

      } /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 5, constant coefficient 0 */
HYPRE_Int
hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   HYPRE_Int             loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;
   hypre_Box            *cgrid_box;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double                a_cw_offd, a_cw_offdm1, a_cw_offdp1, a_ce_offdm1;
   double                a_cs_offd, a_cs_offdm1, a_cs_offdp1, a_cn_offd, a_cn_offdm1;
   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
                      
   HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd; 
   HYPRE_Int             xOffsetP; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

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

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_BoxOffsetDistance(R_dbox, index);
 
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
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    * 
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);
   }
   else
   {
      yOffsetA_offd = hypre_CCBoxOffsetDistance(A_dbox,index);
      yOffsetA_diag = hypre_BoxOffsetDistance(A_dbox,index);
   }

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox,index);


   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA;
            iAp1 = iA + yOffsetA;

            iP1 = iP - yOffsetP - xOffsetP;
            rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

            iP1 = iP - yOffsetP;
            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
               +          rb[iR] * a_cs[iAm1]
               +                   a_cs[iA]   * pa[iP1];

            iP1 = iP - yOffsetP + xOffsetP;
            rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

            iP1 = iP - xOffsetP;
            rap_cw[iAc] =          a_cw[iA]
               +          rb[iR] * a_cw[iAm1] * pb[iP1]
               +          ra[iR] * a_cw[iAp1] * pa[iP1];

            rap_cc[iAc] =          a_cc[iA]
               +          rb[iR] * a_cc[iAm1] * pb[iP]
               +          ra[iR] * a_cc[iAp1] * pa[iP]
               +          rb[iR] * a_cn[iAm1]
               +          ra[iR] * a_cs[iAp1]
               +                   a_cs[iA]   * pb[iP]
               +                   a_cn[iA]   * pa[iP];
         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }
   else
   {
      iA_offd = hypre_CCBoxIndexRank(A_dbox,fstart);
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_ce_offdm1 = a_ce[iA_offdm1];

      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA_diag;
            iAp1 = iA + yOffsetA_diag;

            iP1 = iP - yOffsetP - xOffsetP;
            rap_csw[iAc] = rb[iR] * a_cw_offdm1 * pa[iP1];

            iP1 = iP - yOffsetP;
            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
               +          rb[iR] * a_cs_offdm1
               +                   a_cs_offd   * pa[iP1];

            iP1 = iP - yOffsetP + xOffsetP;
            rap_cse[iAc] = rb[iR] * a_ce_offdm1 * pa[iP1];

            iP1 = iP - xOffsetP;
            rap_cw[iAc] =          a_cw_offd
               +          rb[iR] * a_cw_offdm1 * pb[iP1]
               +          ra[iR] * a_cw_offdp1 * pa[iP1];

            rap_cc[iAc] =          a_cc[iA]
               +          rb[iR] * a_cc[iAm1] * pb[iP]
               +          ra[iR] * a_cc[iAp1] * pa[iP]
               +          rb[iR] * a_cn_offdm1
               +          ra[iR] * a_cs_offdp1
               +                   a_cs_offd  * pb[iP]
               +                   a_cn_offd  * pa[iP];
         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }

/*      } *//* end ForBoxI */

   return ierr;
}

/* for fine stencil size 5, constant coefficient 1 */
HYPRE_Int
hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA; 
   HYPRE_Int             xOffsetP; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

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

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_CCBoxOffsetDistance(R_dbox, index);
 
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
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    * 
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_CCBoxOffsetDistance(P_dbox,index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 9-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   iP = hypre_CCBoxIndexRank(P_dbox,cstart);
   iR = hypre_CCBoxIndexRank(R_dbox,cstart);
   iA = hypre_CCBoxIndexRank(A_dbox,fstart);
   iAc = hypre_CCBoxIndexRank(RAP_dbox,cstart);

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
      +          rb[iR] * a_cs[iAm1]
      +                   a_cs[iA]   * pa[iP1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
      +          rb[iR] * a_cw[iAm1] * pb[iP1]
      +          ra[iR] * a_cw[iAp1] * pa[iP1];

   rap_cc[iAc] =          a_cc[iA]
      +          rb[iR] * a_cc[iAm1] * pb[iP]
      +          ra[iR] * a_cc[iAp1] * pa[iP]
      +          rb[iR] * a_cn[iAm1]
      +          ra[iR] * a_cs[iAp1]
      +                   a_cs[iA]   * pb[iP]
      +                   a_cn[iA]   * pa[iP];

/*      } *//* end ForBoxI */

   return ierr;
}

/* for fine stencil size 9, constant coefficient 0 */
HYPRE_Int
hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   HYPRE_Int             loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *a_csw, *a_cse, *a_cnw;
   double                a_cw_offd, a_cw_offdm1, a_cw_offdp1, a_ce_offdm1;
   double                a_cs_offd, a_cs_offdm1, a_cs_offdp1, a_cn_offd, a_cn_offdm1;
   double                a_csw_offd, a_csw_offdm1, a_csw_offdp1, a_cse_offd, a_cse_offdm1;
   double                a_cnw_offd, a_cnw_offdm1;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
                      
   HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd; 
   HYPRE_Int             xOffsetP; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_BoxOffsetDistance(R_dbox, index);
 
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
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    * 
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);
   }
   else
   {
      yOffsetA_offd = hypre_CCBoxOffsetDistance(A_dbox,index);
      yOffsetA_diag = hypre_BoxOffsetDistance(A_dbox,index);
   }

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox,index);

   /*--------------------------------------------------------------
    * Loop for symmetric 9-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA;
            iAp1 = iA + yOffsetA;

            iP1 = iP - yOffsetP - xOffsetP;
            rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
               +           rb[iR] * a_csw[iAm1]
               +                    a_csw[iA]  * pa[iP1];

            iP1 = iP - yOffsetP;
            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
               +          rb[iR] * a_cs[iAm1]
               +                   a_cs[iA]   * pa[iP1];

            iP1 = iP - yOffsetP + xOffsetP;
            rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
               +           rb[iR] * a_cse[iAm1]
               +                    a_cse[iA]  * pa[iP1];

            iP1 = iP - xOffsetP;
            rap_cw[iAc] =          a_cw[iA]
               +          rb[iR] * a_cw[iAm1] * pb[iP1]
               +          ra[iR] * a_cw[iAp1] * pa[iP1]
               +          rb[iR] * a_cnw[iAm1]
               +          ra[iR] * a_csw[iAp1]
               +                   a_csw[iA]  * pb[iP1]
               +                   a_cnw[iA]  * pa[iP1];

            rap_cc[iAc] =          a_cc[iA]
               +          rb[iR] * a_cc[iAm1] * pb[iP]
               +          ra[iR] * a_cc[iAp1] * pa[iP]
               +          rb[iR] * a_cn[iAm1]
               +          ra[iR] * a_cs[iAp1]
               +                   a_cs[iA]   * pb[iP]
               +                   a_cn[iA]   * pa[iP];

         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }
   else
   {
      iA_offd = hypre_CCBoxIndexRank(A_dbox,fstart);
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_csw_offd = a_csw[iA_offd];
      a_csw_offdm1 = a_csw[iA_offdm1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdm1 = a_cse[iA_offdm1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdm1 = a_cnw[iA_offdm1];


      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA_diag;
            iAp1 = iA + yOffsetA_diag;

            iP1 = iP - yOffsetP - xOffsetP;
            rap_csw[iAc] = rb[iR] * a_cw_offdm1 * pa[iP1]
               +           rb[iR] * a_csw_offdm1
               +                    a_csw_offd  * pa[iP1];

            iP1 = iP - yOffsetP;
            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
               +          rb[iR] * a_cs_offdm1
               +                   a_cs_offd   * pa[iP1];

            iP1 = iP - yOffsetP + xOffsetP;
            rap_cse[iAc] = rb[iR] * a_ce_offdm1 * pa[iP1]
               +           rb[iR] * a_cse_offdm1
               +                    a_cse_offd  * pa[iP1];

            iP1 = iP - xOffsetP;
            rap_cw[iAc] =          a_cw_offd
               +          rb[iR] * a_cw_offdm1 * pb[iP1]
               +          ra[iR] * a_cw_offdp1 * pa[iP1]
               +          rb[iR] * a_cnw_offdm1
               +          ra[iR] * a_csw_offdp1
               +                   a_csw_offd  * pb[iP1]
               +                   a_cnw_offd  * pa[iP1];

            rap_cc[iAc] =          a_cc[iA]
               +          rb[iR] * a_cc[iAm1] * pb[iP]
               +          ra[iR] * a_cc[iAp1] * pa[iP]
               +          rb[iR] * a_cn_offdm1
               +          ra[iR] * a_cs_offdp1
               +                   a_cs_offd   * pb[iP]
               +                   a_cn_offd   * pa[iP];

         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }

/*      }*/ /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 9, constant coefficient 1 */
HYPRE_Int
hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *a_csw, *a_cse, *a_cnw;
   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA; 
   HYPRE_Int             xOffsetP; 
   HYPRE_Int             yOffsetP; 
                      
   HYPRE_Int             ierr = 0;

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_CCBoxOffsetDistance(R_dbox, index);
 
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
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    * 
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_CCBoxOffsetDistance(P_dbox,index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 9-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 9-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   iP = hypre_CCBoxIndexRank(P_dbox,cstart);
   iR = hypre_CCBoxIndexRank(R_dbox,cstart);
   iA = hypre_CCBoxIndexRank(A_dbox,fstart);
   iAc = hypre_CCBoxIndexRank(RAP_dbox,cstart);

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
      +           rb[iR] * a_csw[iAm1]
      +                    a_csw[iA]  * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
      +          rb[iR] * a_cs[iAm1]
      +                   a_cs[iA]   * pa[iP1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
      +           rb[iR] * a_cse[iAm1]
      +                    a_cse[iA]  * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
      +          rb[iR] * a_cw[iAm1] * pb[iP1]
      +          ra[iR] * a_cw[iAp1] * pa[iP1]
      +          rb[iR] * a_cnw[iAm1]
      +          ra[iR] * a_csw[iAp1]
      +                   a_csw[iA]  * pb[iP1]
      +                   a_cnw[iA]  * pa[iP1];

   rap_cc[iAc] =          a_cc[iA]
      +          rb[iR] * a_cc[iAm1] * pb[iP]
      +          ra[iR] * a_cc[iAp1] * pa[iP]
      +          rb[iR] * a_cn[iAm1]
      +          ra[iR] * a_cs[iAp1]
      +                   a_cs[iA]   * pb[iP]
      +                   a_cn[iA]   * pa[iP];

 

/*      }*/ /* end ForBoxI */

   return ierr;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMG2BuildRAPNoSym( hypre_StructMatrix *A,
                          hypre_StructMatrix *P,
                          hypre_StructMatrix *R,
                          HYPRE_Int           cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride,
                          hypre_StructMatrix *RAP     )
{

   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;
                        
   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   HYPRE_Int             fi, ci;
   HYPRE_Int             constant_coefficient;
   HYPRE_Int             constant_coefficient_A;
   HYPRE_Int             ierr = 0;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient)
   {
      hypre_assert( hypre_StructMatrixConstantCoefficient(R) );
      hypre_assert( hypre_StructMatrixConstantCoefficient(A) );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P) );
   }
   else
   {
/*      hypre_assert( hypre_StructMatrixConstantCoefficient(R)==0 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(A)==0 );
      hypre_assert( hypre_StructMatrixConstantCoefficient(P)==0 );
*/
   }

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         /*-----------------------------------------------------------------
          * Switch statement to direct control to appropriate BoxLoop depending
          * on stencil size. Default is full 27-point.
          *-----------------------------------------------------------------*/

         switch (fine_stencil_size)
         {

            /*--------------------------------------------------------------
             * Loop for 5-point fine grid operator; produces upper triangular
             * part of 9-point coarse grid operator - excludes diagonal.
             * stencil entries: (northeast, north, northwest, and east)
             *--------------------------------------------------------------*/

         case 5:

            if ( constant_coefficient==1 )
            {
               ierr += hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               ierr += hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

            /*--------------------------------------------------------------
             * Loop for 9-point fine grid operator; produces upper triangular
             * part of 9-point coarse grid operator - excludes diagonal.
             * stencil entries: (northeast, north, northwest, and east)
             *--------------------------------------------------------------*/

         default:

            if ( constant_coefficient==1 )
            {
               ierr += hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               ierr += hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         } /* end switch statement */

      } /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 5, constant coefficient 0 */
HYPRE_Int
hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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
                        
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             loopi, loopj, loopk;
   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cn;
   double               a_cn_offd, a_cn_offdp1, a_cw_offdp1;
   double               a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   HYPRE_Int             iA, iAm1, iAp1, iA_offd, iA_offdm1, iA_offdp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
                     
   HYPRE_Int             ierr = 0;

   /*hypre_printf("nosym 5.0\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_BoxOffsetDistance(R_dbox, index);
 
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

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);
   }
   else
   {
      hypre_assert( constant_coefficient_A==2 );
      yOffsetA_diag = hypre_BoxOffsetDistance(A_dbox,index);
      yOffsetA_offd = hypre_CCBoxOffsetDistance(A_dbox,index);
   }

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox,index);


   /*--------------------------------------------------------------
    * Loop for 5-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
      /*hypre_printf("nosym 5.0.0\n");*/
      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA;
            iAp1 = iA + yOffsetA;

            iP1 = iP + yOffsetP + xOffsetP;
            rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

            iP1 = iP + yOffsetP;
            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
               +          ra[iR] * a_cn[iAp1]
               +                   a_cn[iA]   * pb[iP1];

            iP1 = iP + yOffsetP - xOffsetP;
            rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

            iP1 = iP + xOffsetP;
            rap_ce[iAc] =          a_ce[iA]
               +          rb[iR] * a_ce[iAm1] * pb[iP1]
               +          ra[iR] * a_ce[iAp1] * pa[iP1];
         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }
   else
   {
      hypre_assert( constant_coefficient_A==2 );
      /*hypre_printf("nosym 5.0.2\n"); */

      iA_offd = hypre_CCBoxIndexRank(A_dbox,fstart);
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];

      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAp1 = iA + yOffsetA_diag;

            iP1 = iP + yOffsetP + xOffsetP;
            rap_cne[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1];

            iP1 = iP + yOffsetP;
            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
               +          ra[iR] * a_cn_offdp1
               +                   a_cn_offd   * pb[iP1];

            iP1 = iP + yOffsetP - xOffsetP;
            rap_cnw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1];

            iP1 = iP + xOffsetP;
            rap_ce[iAc] =          a_ce_offd
               +          rb[iR] * a_ce_offdm1 * pb[iP1]
               +          ra[iR] * a_ce_offdp1 * pa[iP1];
         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }

/*      }*/ /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 5, constant coefficient 1 */
HYPRE_Int
hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;
   double               *a_cc, *a_cw, *a_ce, *a_cn;
   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
                     
   HYPRE_Int             ierr = 0;

   /* hypre_printf("nosym 5.1\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_CCBoxOffsetDistance(R_dbox, index);
 
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

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_CCBoxOffsetDistance(P_dbox,index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 5-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   iP = hypre_CCBoxIndexRank(P_dbox,cstart);
   iR = hypre_CCBoxIndexRank(R_dbox,cstart);
   iA = hypre_CCBoxIndexRank(A_dbox,fstart);
   iAc = hypre_CCBoxIndexRank(RAP_dbox,cstart);

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
      +          ra[iR] * a_cn[iAp1]
      +                   a_cn[iA]   * pb[iP1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
      +          rb[iR] * a_ce[iAm1] * pb[iP1]
      +          ra[iR] * a_ce[iAp1] * pa[iP1];


/*      }*/ /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 9, constant coefficient 0 */
HYPRE_Int
hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             loopi, loopj, loopk;
   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;
   double               *a_cc, *a_cw, *a_ce, *a_cn;
   double               *a_cse, *a_cnw, *a_cne;
   double               a_cn_offd, a_cn_offdp1, a_cw_offdp1;
   double               a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   double               a_cne_offd, a_cne_offdm1, a_cne_offdp1;
   double               a_cse_offd, a_cse_offdp1, a_cnw_offd, a_cnw_offdp1;
   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   HYPRE_Int             iA, iAm1, iAp1, iA_offd, iA_offdm1, iA_offdp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
                     
   HYPRE_Int             ierr = 0;

   /*hypre_printf("nosym 9.0\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_BoxOffsetDistance(R_dbox, index);
 
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

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = hypre_BoxOffsetDistance(A_dbox,index);
   }
   else
   {
      hypre_assert( constant_coefficient_A==2 );
      yOffsetA_diag = hypre_BoxOffsetDistance(A_dbox,index);
      yOffsetA_offd = hypre_CCBoxOffsetDistance(A_dbox,index);
   }

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox,index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/


   /*--------------------------------------------------------------
    * Loop for 9-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A==0 )
   {
      /*hypre_printf("nosym 9.0.0\n");*/
      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA;
            iAp1 = iA + yOffsetA;

            iP1 = iP + yOffsetP + xOffsetP;
            rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
               +           ra[iR] * a_cne[iAp1]
               +                    a_cne[iA]  * pb[iP1];

            iP1 = iP + yOffsetP;
            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
               +          ra[iR] * a_cn[iAp1]
               +                   a_cn[iA]   * pb[iP1];

            iP1 = iP + yOffsetP - xOffsetP;
            rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
               +           ra[iR] * a_cnw[iAp1]
               +                    a_cnw[iA]  * pb[iP1];

            iP1 = iP + xOffsetP;
            rap_ce[iAc] =          a_ce[iA]
               +          rb[iR] * a_ce[iAm1] * pb[iP1]
               +          ra[iR] * a_ce[iAp1] * pa[iP1]
               +          rb[iR] * a_cne[iAm1]
               +          ra[iR] * a_cse[iAp1]
               +                   a_cse[iA]  * pb[iP1]
               +                   a_cne[iA]  * pa[iP1];

         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }
   else
   {
      /*hypre_printf("nosym 9.0.2\n");*/
      hypre_assert( constant_coefficient_A==2 );
      iA_offd = hypre_CCBoxIndexRank(A_dbox,fstart);
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];
      a_cne_offd = a_cne[iA_offd];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cne_offdp1 = a_cne[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdp1 = a_cnw[iA_offdp1];

      hypre_BoxLoop4Begin(loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAm1,iAp1,iP1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
         {
            iAm1 = iA - yOffsetA_diag;
            iAp1 = iA + yOffsetA_diag;

            iP1 = iP + yOffsetP + xOffsetP;
            rap_cne[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1]
               +           ra[iR] * a_cne_offdp1
               +                    a_cne_offd  * pb[iP1];

            iP1 = iP + yOffsetP;
            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
               +          ra[iR] * a_cn_offdp1
               +                   a_cn_offd   * pb[iP1];

            iP1 = iP + yOffsetP - xOffsetP;
            rap_cnw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1]
               +           ra[iR] * a_cnw_offdp1
               +                    a_cnw_offd  * pb[iP1];

            iP1 = iP + xOffsetP;
            rap_ce[iAc] =          a_ce_offd
               +          rb[iR] * a_ce_offdm1 * pb[iP1]
               +          ra[iR] * a_ce_offdp1 * pa[iP1]
               +          rb[iR] * a_cne_offdm1
               +          ra[iR] * a_cse_offdp1
               +                   a_cse_offd  * pb[iP1]
               +                   a_cne_offd  * pa[iP1];

         }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
   }

/*      }*/ /* end ForBoxI */

   return ierr;
}

/* for fine stencil size 9, constant coefficient 1 */
HYPRE_Int
hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
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

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;
   double               *a_cc, *a_cw, *a_ce, *a_cn;
   double               *a_cse, *a_cnw, *a_cne;
   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             yOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
                     
   HYPRE_Int             ierr = 0;

   /*hypre_printf("nosym 9.1\n");*/
   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

/*   fi = 0;
     hypre_ForBoxI(ci, cgrid_boxes)
     {
     while (fgrid_ids[fi] != cgrid_ids[ci])
     {
     fi++;
     }
*/
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
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
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point 
    * rb is pointer for weight for f-point below c-point 
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,0,-1,0);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
      hypre_CCBoxOffsetDistance(R_dbox, index);
 
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

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,-1,0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,0,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex(index_temp,-1,1,0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

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

   hypre_SetIndex(index_temp,1,0,0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_CCBoxOffsetDistance(P_dbox,index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/


   /*--------------------------------------------------------------
    * Loop for 9-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   iP = hypre_CCBoxIndexRank(P_dbox,cstart);
   iR = hypre_CCBoxIndexRank(R_dbox,cstart);
   iA = hypre_CCBoxIndexRank(A_dbox,fstart);
   iAc = hypre_CCBoxIndexRank(RAP_dbox,cstart);

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
      +           ra[iR] * a_cne[iAp1]
      +                    a_cne[iA]  * pb[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
      +          ra[iR] * a_cn[iAp1]
      +                   a_cn[iA]   * pb[iP1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
      +           ra[iR] * a_cnw[iAp1]
      +                    a_cnw[iA]  * pb[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
      +          rb[iR] * a_ce[iAm1] * pb[iP1]
      +          ra[iR] * a_ce[iAp1] * pa[iP1]
      +          rb[iR] * a_cne[iAm1]
      +          ra[iR] * a_cse[iAp1]
      +                   a_cse[iA]  * pb[iP1]
      +                   a_cne[iA]  * pa[iP1];



/*      }*/ /* end ForBoxI */

   return ierr;
}




/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "smg.h"

/*--------------------------------------------------------------------------
 * Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SMG3CreateRAPOp( hypre_StructMatrix *R,
                       hypre_StructMatrix *A,
                       hypre_StructMatrix *PT,
                       hypre_StructGrid   *coarse_grid )
{
   HYPRE_UNUSED_VAR(R);
   HYPRE_UNUSED_VAR(PT);

   hypre_StructMatrix    *RAP;

   hypre_Index           *RAP_stencil_shape;
   hypre_StructStencil   *RAP_stencil;
   HYPRE_Int              RAP_stencil_size;
   HYPRE_Int              RAP_stencil_dim;
   HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   hypre_StructStencil   *A_stencil;
   HYPRE_Int              A_stencil_size;

   HYPRE_Int              k, j, i;
   HYPRE_Int              stencil_rank;

   RAP_stencil_dim = 3;

   A_stencil = hypre_StructMatrixStencil(A);
   A_stencil_size = hypre_StructStencilSize(A_stencil);

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
       * 7 or 15 point fine grid stencil produces 15 point RAP
       *--------------------------------------------------------------------*/
      if ( A_stencil_size <= 15)
      {
         RAP_stencil_size = 15;
         RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
         for (k = -1; k < 2; k++)
         {
            for (j = -1; j < 2; j++)
            {
               for (i = -1; i < 2; i++)
               {

                  /*--------------------------------------------------------
                   * Storage for c,w,e,n,s elements in each plane
                   *--------------------------------------------------------*/
                  if ( i * j == 0 )
                  {
                     hypre_SetIndex3(RAP_stencil_shape[stencil_rank], i, j, k);
                     stencil_rank++;
                  }
               }
            }
         }
      }

      /*--------------------------------------------------------------------
       * 19 or 27 point fine grid stencil produces 27 point RAP
       *--------------------------------------------------------------------*/
      else
      {
         RAP_stencil_size = 27;
         RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
         for (k = -1; k < 2; k++)
         {
            for (j = -1; j < 2; j++)
            {
               for (i = -1; i < 2; i++)
               {

                  /*--------------------------------------------------------
                   * Storage for 9 elements (c,w,e,n,s,sw,se,nw,se) in
                   * each plane
                   *--------------------------------------------------------*/
                  hypre_SetIndex3(RAP_stencil_shape[stencil_rank], i, j, k);
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
       * 7 or 15 point fine grid stencil produces 15 point RAP
       * Only store the lower triangular part + diagonal = 8 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicalgraphic ordering.
       *--------------------------------------------------------------------*/
      if ( A_stencil_size <= 15)
      {
         RAP_stencil_size = 8;
         RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
         for (k = -1; k < 1; k++)
         {
            for (j = -1; j < 2; j++)
            {
               for (i = -1; i < 2; i++)
               {

                  /*--------------------------------------------------------
                   * Store  5 elements in lower plane (c,w,e,s,n)
                   * and 3 elements in same plane (c,w,s)
                   *--------------------------------------------------------*/
                  if ( i * j == 0 && i + j + k <= 0)
                  {
                     hypre_SetIndex3(RAP_stencil_shape[stencil_rank], i, j, k);
                     stencil_rank++;
                  }
               }
            }
         }
      }

      /*--------------------------------------------------------------------
       * 19 or 27 point fine grid stencil produces 27 point RAP
       * Only store the lower triangular part + diagonal = 14 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicalgraphic ordering.
       *--------------------------------------------------------------------*/
      else
      {
         RAP_stencil_size = 14;
         RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
         for (k = -1; k < 1; k++)
         {
            for (j = -1; j < 2; j++)
            {
               for (i = -1; i < 2; i++)
               {

                  /*--------------------------------------------------------
                   * Store  9 elements in lower plane (c,w,e,s,n,sw,se,nw,ne)
                   * and 5 elements in same plane (c,w,s,sw,se)
                   *--------------------------------------------------------*/
                  if ( k < 0 || (i + j + k <= 0 && j < 1) )
                  {
                     hypre_SetIndex3(RAP_stencil_shape[stencil_rank], i, j, k);
                     stencil_rank++;
                  }
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
    * Set number of ghost points
    *-----------------------------------------------------------------------*/
   if (hypre_StructMatrixSymmetric(A))
   {
      RAP_num_ghost[1] = 0;
      RAP_num_ghost[3] = 0;
      RAP_num_ghost[5] = 0;
   }
   hypre_StructMatrixSetNumGhost(RAP, RAP_num_ghost);

   return RAP;
}

/*--------------------------------------------------------------------------
 * Routines to build RAP. These routines are fairly general
 *  1) No assumptions about symmetry of A
 *  2) No assumption that R = transpose(P)
 *  3) 7,15,19 or 27-point fine grid A
 *
 * I am, however, assuming that the c-to-c interpolation is the identity.
 *
 * I've written a two routines - hypre_SMG3BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * hypre_SMG3BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMG3BuildRAPSym( hypre_StructMatrix *A,
                       hypre_StructMatrix *PT,
                       hypre_StructMatrix *R,
                       hypre_StructMatrix *RAP,
                       hypre_Index         cindex,
                       hypre_Index         cstride )

{

   hypre_Index           index;

   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;

   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             fi, ci;

   hypre_Box            *A_dbox;
   hypre_Box            *PT_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_bc;
   HYPRE_Real           *a_aw  = NULL, *a_as  = NULL;
   HYPRE_Real           *a_bw  = NULL, *a_be  = NULL;
   HYPRE_Real           *a_bs  = NULL, *a_bn  = NULL;
   HYPRE_Real           *a_csw = NULL, *a_cse = NULL;
   HYPRE_Real           *a_cnw = NULL, *a_cne = NULL;
   HYPRE_Real           *a_asw = NULL, *a_ase = NULL;
   HYPRE_Real           *a_bsw = NULL, *a_bse = NULL;
   HYPRE_Real           *a_bnw = NULL, *a_bne = NULL;

   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw = NULL, *rap_cse = NULL;
   HYPRE_Real           *rap_bsw = NULL, *rap_bse = NULL;
   HYPRE_Real           *rap_bnw = NULL, *rap_bne = NULL;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
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
      PT_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(PT), fi);
      R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
      RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pa is pointer for weight for f-point above c-point
       * pb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      pa = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);

      hypre_SetIndex3(index, 0, 0, -1);
      pb = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);

      /*-----------------------------------------------------------------
       * Extract pointers for restriction operator:
       * ra is pointer for weight for f-point above c-point
       * rb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

      hypre_SetIndex3(index, 0, 0, -1);
      rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

      /*-----------------------------------------------------------------
       * Extract pointers for 7-point fine grid operator:
       *
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient in same plane
       * a_ce is pointer for east coefficient in same plane
       * a_cs is pointer for south coefficient in same plane
       * a_cn is pointer for north coefficient in same plane
       * a_ac is pointer for center coefficient in plane above
       * a_bc is pointer for center coefficient in plane below
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, -1, 0, 0);
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 1, 0, 0);
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, -1, 0);
      a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, 1, 0);
      a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, 0, 1);
      a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, 0, -1);
      a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      /*-----------------------------------------------------------------
       * Extract additional pointers for 15-point fine grid operator:
       *
       * a_aw is pointer for west coefficient in plane above
       * a_ae is pointer for east coefficient in plane above
       * a_as is pointer for south coefficient in plane above
       * a_an is pointer for north coefficient in plane above
       * a_bw is pointer for west coefficient in plane below
       * a_be is pointer for east coefficient in plane below
       * a_bs is pointer for south coefficient in plane below
       * a_bn is pointer for north coefficient in plane below
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 7)
      {
         hypre_SetIndex3(index, -1, 0, 1);
         a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, -1, 1);
         a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 0, -1);
         a_bw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 0, -1);
         a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, -1, -1);
         a_bs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, 1, -1);
         a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      }

      /*-----------------------------------------------------------------
       * Extract additional pointers for 19-point fine grid operator:
       *
       * a_csw is pointer for southwest coefficient in same plane
       * a_cse is pointer for southeast coefficient in same plane
       * a_cnw is pointer for northwest coefficient in same plane
       * a_cne is pointer for northeast coefficient in same plane
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 15)
      {
         hypre_SetIndex3(index, -1, -1, 0);
         a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, -1, 0);
         a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 1, 0);
         a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 1, 0);
         a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract additional pointers for 27-point fine grid operator:
       *
       * a_asw is pointer for southwest coefficient in plane above
       * a_ase is pointer for southeast coefficient in plane above
       * a_anw is pointer for northwest coefficient in plane above
       * a_ane is pointer for northeast coefficient in plane above
       * a_bsw is pointer for southwest coefficient in plane below
       * a_bse is pointer for southeast coefficient in plane below
       * a_bnw is pointer for northwest coefficient in plane below
       * a_bne is pointer for northeast coefficient in plane below
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 19)
      {
         hypre_SetIndex3(index, -1, -1, 1);
         a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, -1, 1);
         a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, -1, -1);
         a_bsw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, -1, -1);
         a_bse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 1, -1);
         a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 1, -1);
         a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract pointers for 15-point coarse grid operator:
       *
       * We build only the lower triangular part (plus diagonal).
       *
       * rap_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, -1, 0, 0);
      rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, -1, 0);
      rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, 0, -1);
      rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, -1, 0, -1);
      rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 1, 0, -1);
      rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, -1, -1);
      rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, 1, -1);
      rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      /*-----------------------------------------------------------------
       * Extract additional pointers for 27-point coarse grid operator:
       *
       * A 27-point coarse grid operator is produced when the fine grid
       * stencil is 19 or 27 point.
       *
       * We build only the lower triangular part.
       *
       * rap_csw is pointer for southwest coefficient in same plane (etc.)
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 15)
      {
         hypre_SetIndex3(index, -1, -1, 0);
         rap_csw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, -1, 0);
         rap_cse =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, -1, -1);
         rap_bsw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, -1, -1);
         rap_bse =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 1, -1);
         rap_bnw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 1, -1);
         rap_bne =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      }

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets are used in refering to data associated with other points.
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);
      hypre_SetIndex3(index, 0, 1, 0);
      yOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);
      hypre_SetIndex3(index, 1, 0, 0);
      xOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);

      /*--------------------------------------------------------------------
       * Switch statement to direct control to apropriate BoxLoop depending
       * on stencil size. Default is full 27-point.
       *-----------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for symmetric 7-point fine grid operator; produces a
          * symmetric 15-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-south, below-west,
          * below-center, below-east, below-north, center-south,
          * center-west, and center-center).
          *--------------------------------------------------------------*/

         case 7:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_bs,rb,a_cs,pa,rap_bw,a_cw,rap_bc,a_bc,a_cc,rap_be,a_ce,rap_bn,a_cn,rap_cs,pb,ra,rap_cw,rap_cc,a_ac)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP - zOffsetP - yOffsetP;
               rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP - xOffsetP;
               rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP;
               rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                                      +          rb[iR] * a_cc[iAm1] * pa[iP1]
                                      +          rb[iR] * a_bc[iAm1];

               iP1 = iP - zOffsetP + xOffsetP;
               rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP;
               rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1];

               iP1 = iP - yOffsetP;
               rap_cs[iAc] =          a_cs[iA]
                                      +          rb[iR] * a_cs[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cs[iAp1] * pa[iP1];

               iP1 = iP - xOffsetP;
               rap_cw[iAc] =          a_cw[iA]
                                      +          rb[iR] * a_cw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cw[iAp1] * pa[iP1];

               rap_cc[iAc] =          a_cc[iA]
                                      +          rb[iR] * a_cc[iAm1] * pb[iP]
                                      +          ra[iR] * a_cc[iAp1] * pa[iP]
                                      +          rb[iR] * a_ac[iAm1]
                                      +          ra[iR] * a_bc[iAp1]
                                      +                   a_bc[iA]   * pb[iP]
                                      +                   a_ac[iA]   * pa[iP];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 15-point fine grid operator; produces a
          * symmetric 15-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-south, below-west,
          * below-center, below-east, below-north, center-south,
          * center-west, and center-center).
          *--------------------------------------------------------------*/

         case 15:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_bs,rb,a_cs,pa,a_bs,rap_bw,a_cw,a_bw,rap_bc,a_bc,a_cc,rap_be,a_ce,a_be,rap_bn,a_cn,a_bn,rap_cs,pb,ra,a_as,rap_cw,a_aw,rap_cc,a_ac)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP - zOffsetP - yOffsetP;
               rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                             +          rb[iR] * a_bs[iAm1]
                             +                   a_bs[iA]   * pa[iP1];

               iP1 = iP - zOffsetP - xOffsetP;
               rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                             +          rb[iR] * a_bw[iAm1]
                             +                   a_bw[iA]   * pa[iP1];

               iP1 = iP - zOffsetP;
               rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                                      +          rb[iR] * a_cc[iAm1] * pa[iP1]
                                      +          rb[iR] * a_bc[iAm1];

               iP1 = iP - zOffsetP + xOffsetP;
               rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                             +          rb[iR] * a_be[iAm1]
                             +                   a_be[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP;
               rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                             +          rb[iR] * a_bn[iAm1]
                             +                   a_bn[iA]   * pa[iP1];

               iP1 = iP - yOffsetP;
               rap_cs[iAc] =          a_cs[iA]
                                      +          rb[iR] * a_cs[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cs[iAp1] * pa[iP1]
                                      +                   a_bs[iA]   * pb[iP1]
                                      +                   a_as[iA]   * pa[iP1]
                                      +          rb[iR] * a_as[iAm1]
                                      +          ra[iR] * a_bs[iAp1];

               iP1 = iP - xOffsetP;
               rap_cw[iAc] =          a_cw[iA]
                                      +          rb[iR] * a_cw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                      +                   a_bw[iA]   * pb[iP1]
                                      +                   a_aw[iA]   * pa[iP1]
                                      +          rb[iR] * a_aw[iAm1]
                                      +          ra[iR] * a_bw[iAp1];

               rap_cc[iAc] =          a_cc[iA]
                                      +          rb[iR] * a_cc[iAm1] * pb[iP]
                                      +          ra[iR] * a_cc[iAp1] * pa[iP]
                                      +          rb[iR] * a_ac[iAm1]
                                      +          ra[iR] * a_bc[iAp1]
                                      +                   a_bc[iA]   * pb[iP]
                                      +                   a_ac[iA]   * pa[iP];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 19-point fine grid operator; produces a
          * symmetric 27-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-southwest, below-south,
          * below-southeast, below-west, below-center, below-east,
          * below-northwest, below-north, below-northeast, center-southwest,
          * center-south, center-southeast, center-west, and center-center).
          *--------------------------------------------------------------*/

         case 19:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_bsw,rb,a_csw,pa,rap_bs,a_cs,a_bs,rap_bse,a_cse,rap_bw,a_cw,a_bw,rap_bc,a_bc,a_cc,rap_be,a_ce,a_be,rap_bnw,a_cnw,rap_bn,a_cn,a_bn,rap_bne,a_cne,rap_csw,pb,ra,rap_cs,a_as,rap_cse,rap_cw,a_aw,rap_cc,a_ac)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
               rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP - yOffsetP;
               rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                             +          rb[iR] * a_bs[iAm1]
                             +                   a_bs[iA]   * pa[iP1];

               iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
               rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP - xOffsetP;
               rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                             +          rb[iR] * a_bw[iAm1]
                             +                   a_bw[iA]   * pa[iP1];

               iP1 = iP - zOffsetP;
               rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                      +          rb[iR] * a_cc[iAm1] * pa[iP1]
                                      +          rb[iR] * a_bc[iAm1];

               iP1 = iP - zOffsetP + xOffsetP;
               rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                             +          rb[iR] * a_be[iAm1]
                             +                   a_be[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
               rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP;
               rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                             +          rb[iR] * a_bn[iAm1]
                             +                   a_bn[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
               rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1];

               iP1 = iP - yOffsetP - xOffsetP;
               rap_csw[iAc] =         a_csw[iA]
                                      +          rb[iR] * a_csw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_csw[iAp1] * pa[iP1];

               iP1 = iP - yOffsetP;
               rap_cs[iAc] =          a_cs[iA]
                                      +          rb[iR] * a_cs[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cs[iAp1] * pa[iP1]
                                      +                   a_bs[iA]   * pb[iP1]
                                      +                   a_as[iA]   * pa[iP1]
                                      +          rb[iR] * a_as[iAm1]
                                      +          ra[iR] * a_bs[iAp1];

               iP1 = iP - yOffsetP + xOffsetP;
               rap_cse[iAc] =          a_cse[iA]
                                       +          rb[iR] * a_cse[iAm1] * pb[iP1]
                                       +          ra[iR] * a_cse[iAp1] * pa[iP1];

               iP1 = iP - xOffsetP;
               rap_cw[iAc] =          a_cw[iA]
                                      +          rb[iR] * a_cw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                      +                   a_bw[iA]   * pb[iP1]
                                      +                   a_aw[iA]   * pa[iP1]
                                      +          rb[iR] * a_aw[iAm1]
                                      +          ra[iR] * a_bw[iAp1];

               rap_cc[iAc] =          a_cc[iA]
                                      +          rb[iR] * a_cc[iAm1] * pb[iP]
                                      +          ra[iR] * a_cc[iAp1] * pa[iP]
                                      +          rb[iR] * a_ac[iAm1]
                                      +          ra[iR] * a_bc[iAp1]
                                      +                   a_bc[iA]   * pb[iP]
                                      +                   a_ac[iA]   * pa[iP];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 27-point fine grid operator; produces a
          * symmetric 27-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-southwest, below-south,
          * below-southeast, below-west, below-center, below-east,
          * below-northwest, below-north, below-northeast, center-southwest,
          * center-south, center-southeast, center-west, and center-center).
          *--------------------------------------------------------------*/

         default:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_bsw,rb,a_csw,pa,a_bsw,rap_bs,a_cs,a_bs,rap_bse,a_cse,a_bse,rap_bw,a_cw,a_bw,rap_bc,a_bc,a_cc,rap_be,a_ce,a_be,rap_bnw,a_cnw,a_bnw,rap_bn,a_cn,a_bn,rap_bne,a_cne,a_bne,rap_csw,pb,ra,a_asw,rap_cs,a_as,rap_cse,a_ase,rap_cw,a_aw,rap_cc,a_ac)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
               rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1]
                              +           rb[iR] * a_bsw[iAm1]
                              +                    a_bsw[iA]   * pa[iP1];

               iP1 = iP - zOffsetP - yOffsetP;
               rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                             +          rb[iR] * a_bs[iAm1]
                             +                   a_bs[iA]   * pa[iP1];

               iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
               rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1]
                              +           rb[iR] * a_bse[iAm1]
                              +                    a_bse[iA]   * pa[iP1];

               iP1 = iP - zOffsetP - xOffsetP;
               rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                             +          rb[iR] * a_bw[iAm1]
                             +                   a_bw[iA]   * pa[iP1];

               iP1 = iP - zOffsetP;
               rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                      +          rb[iR] * a_cc[iAm1] * pa[iP1]
                                      +          rb[iR] * a_bc[iAm1];

               iP1 = iP - zOffsetP + xOffsetP;
               rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                             +          rb[iR] * a_be[iAm1]
                             +                   a_be[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
               rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1]
                              +           rb[iR] * a_bnw[iAm1]
                              +                    a_bnw[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP;
               rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                             +          rb[iR] * a_bn[iAm1]
                             +                   a_bn[iA]   * pa[iP1];

               iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
               rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1]
                              +           rb[iR] * a_bne[iAm1]
                              +                    a_bne[iA]   * pa[iP1];

               iP1 = iP - yOffsetP - xOffsetP;
               rap_csw[iAc] =          a_csw[iA]
                                       +          rb[iR] * a_csw[iAm1] * pb[iP1]
                                       +          ra[iR] * a_csw[iAp1] * pa[iP1]
                                       +                   a_bsw[iA]   * pb[iP1]
                                       +                   a_asw[iA]   * pa[iP1]
                                       +          rb[iR] * a_asw[iAm1]
                                       +          ra[iR] * a_bsw[iAp1];

               iP1 = iP - yOffsetP;
               rap_cs[iAc] =          a_cs[iA]
                                      +          rb[iR] * a_cs[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cs[iAp1] * pa[iP1]
                                      +                   a_bs[iA]   * pb[iP1]
                                      +                   a_as[iA]   * pa[iP1]
                                      +          rb[iR] * a_as[iAm1]
                                      +          ra[iR] * a_bs[iAp1];

               iP1 = iP - yOffsetP + xOffsetP;
               rap_cse[iAc] =          a_cse[iA]
                                       +          rb[iR] * a_cse[iAm1] * pb[iP1]
                                       +          ra[iR] * a_cse[iAp1] * pa[iP1]
                                       +                   a_bse[iA]   * pb[iP1]
                                       +                   a_ase[iA]   * pa[iP1]
                                       +          rb[iR] * a_ase[iAm1]
                                       +          ra[iR] * a_bse[iAp1];

               iP1 = iP - xOffsetP;
               rap_cw[iAc] =          a_cw[iA]
                                      +          rb[iR] * a_cw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                      +                   a_bw[iA]   * pb[iP1]
                                      +                   a_aw[iA]   * pa[iP1]
                                      +          rb[iR] * a_aw[iAm1]
                                      +          ra[iR] * a_bw[iAp1];

               rap_cc[iAc] =          a_cc[iA]
                                      +          rb[iR] * a_cc[iAm1] * pb[iP]
                                      +          ra[iR] * a_cc[iAp1] * pa[iP]
                                      +          rb[iR] * a_ac[iAm1]
                                      +          ra[iR] * a_bc[iAp1]
                                      +                   a_bc[iA]   * pb[iP]
                                      +                   a_ac[iA]   * pa[iP];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMG3BuildRAPNoSym( hypre_StructMatrix *A,
                         hypre_StructMatrix *PT,
                         hypre_StructMatrix *R,
                         hypre_StructMatrix *RAP,
                         hypre_Index         cindex,
                         hypre_Index         cstride )

{

   hypre_Index           index;

   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;

   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             fi, ci;

   hypre_Box            *A_dbox;
   hypre_Box            *PT_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw = NULL, *a_ae = NULL;
   HYPRE_Real           *a_as  = NULL, *a_an  = NULL;
   HYPRE_Real           *a_be  = NULL, *a_bn  = NULL;
   HYPRE_Real           *a_csw = NULL, *a_cse = NULL;
   HYPRE_Real           *a_cnw = NULL, *a_cne = NULL;
   HYPRE_Real           *a_asw = NULL, *a_ase = NULL;
   HYPRE_Real           *a_anw = NULL, *a_ane = NULL;
   HYPRE_Real           *a_bnw = NULL, *a_bne = NULL;

   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw = NULL, *rap_cne = NULL;
   HYPRE_Real           *rap_asw = NULL, *rap_ase = NULL;
   HYPRE_Real           *rap_anw = NULL, *rap_ane = NULL;

   HYPRE_Int            zOffsetA;
   HYPRE_Int            xOffsetP;
   HYPRE_Int            yOffsetP;
   HYPRE_Int            zOffsetP;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
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
      PT_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(PT), fi);
      R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
      RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pa is pointer for weight for f-point above c-point
       * pb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      pa = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);

      hypre_SetIndex3(index, 0, 0, -1);
      pb = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);


      /*-----------------------------------------------------------------
       * Extract pointers for restriction operator:
       * ra is pointer for weight for f-point above c-point
       * rb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

      hypre_SetIndex3(index, 0, 0, -1);
      rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);


      /*-----------------------------------------------------------------
       * Extract pointers for 7-point fine grid operator:
       *
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient in same plane
       * a_ce is pointer for east coefficient in same plane
       * a_cs is pointer for south coefficient in same plane
       * a_cn is pointer for north coefficient in same plane
       * a_ac is pointer for center coefficient in plane above
       * a_bc is pointer for center coefficient in plane below
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 0);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, -1, 0, 0);
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 1, 0, 0);
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, -1, 0);
      a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, 1, 0);
      a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index, 0, 0, 1);
      a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      /*-----------------------------------------------------------------
       * Extract additional pointers for 15-point fine grid operator:
       *
       * a_aw is pointer for west coefficient in plane above
       * a_ae is pointer for east coefficient in plane above
       * a_as is pointer for south coefficient in plane above
       * a_an is pointer for north coefficient in plane above
       * a_bw is pointer for west coefficient in plane below
       * a_be is pointer for east coefficient in plane below
       * a_bs is pointer for south coefficient in plane below
       * a_bn is pointer for north coefficient in plane below
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 7)
      {
         hypre_SetIndex3(index, -1, 0, 1);
         a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 0, 1);
         a_ae = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, -1, 1);
         a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, 1, 1);
         a_an = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 0, -1);
         a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 0, 1, -1);
         a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract additional pointers for 19-point fine grid operator:
       *
       * a_csw is pointer for southwest coefficient in same plane
       * a_cse is pointer for southeast coefficient in same plane
       * a_cnw is pointer for northwest coefficient in same plane
       * a_cne is pointer for northeast coefficient in same plane
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 15)
      {
         hypre_SetIndex3(index, -1, -1, 0);
         a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, -1, 0);
         a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 1, 0);
         a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 1, 0);
         a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract additional pointers for 27-point fine grid operator:
       *
       * a_asw is pointer for southwest coefficient in plane above
       * a_ase is pointer for southeast coefficient in plane above
       * a_anw is pointer for northwest coefficient in plane above
       * a_ane is pointer for northeast coefficient in plane above
       * a_bsw is pointer for southwest coefficient in plane below
       * a_bse is pointer for southeast coefficient in plane below
       * a_bnw is pointer for northwest coefficient in plane below
       * a_bne is pointer for northeast coefficient in plane below
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 19)
      {
         hypre_SetIndex3(index, -1, -1, 1);
         a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, -1, 1);
         a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 1, 1);
         a_anw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 1, 1);
         a_ane = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, -1, 1, -1);
         a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index, 1, 1, -1);
         a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract pointers for 15-point coarse grid operator:
       *
       * We build only the upper triangular part (excluding diagonal).
       *
       * rap_ce is pointer for east coefficient in same plane (etc.)
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 1, 0, 0);
      rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, 1, 0);
      rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, 0, 1);
      rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, -1, 0, 1);
      rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 1, 0, 1);
      rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, -1, 1);
      rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index, 0, 1, 1);
      rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      /*-----------------------------------------------------------------
       * Extract additional pointers for 27-point coarse grid operator:
       *
       * A 27-point coarse grid operator is produced when the fine grid
       * stencil is 19 or 27 point.
       *
       * We build only the upper triangular part.
       *
       * rap_cnw is pointer for northwest coefficient in same plane (etc.)
       *-----------------------------------------------------------------*/

      if (fine_stencil_size > 15)
      {
         hypre_SetIndex3(index, -1, 1, 0);
         rap_cnw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 1, 0);
         rap_cne =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, -1, 1);
         rap_asw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, -1, 1);
         rap_ase =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 1, 1);
         rap_anw =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 1, 1);
         rap_ane =
            hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      }

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets are used in refering to data associated with other points.
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index, 0, 0, 1);
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);
      hypre_SetIndex3(index, 0, 1, 0);
      yOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);
      hypre_SetIndex3(index, 1, 0, 0);
      xOffsetP = hypre_BoxOffsetDistance(PT_dbox, index);

      /*-----------------------------------------------------------------
       * Switch statement to direct control to apropriate BoxLoop depending
       * on stencil size. Default is full 27-point.
       *-----------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for 7-point fine grid operator; produces upper triangular
          * part of 15-point coarse grid operator. stencil entries:
          * (above-north, above-east, above-center, above-west,
          * above-south, center-north, and center-east).
          *--------------------------------------------------------------*/

         case 7:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_an,ra,a_cn,pb,rap_ae,a_ce,rap_ac,a_ac,a_cc,rap_aw,a_cw,rap_as,a_cs,rap_cn,rb,pa,rap_ce)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP + zOffsetP + yOffsetP;
               rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP + xOffsetP;
               rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP;
               rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                                      +          ra[iR] * a_cc[iAp1] * pb[iP1]
                                      +          ra[iR] * a_ac[iAp1];

               iP1 = iP + zOffsetP - xOffsetP;
               rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP;
               rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1];

               iP1 = iP + yOffsetP;
               rap_cn[iAc] =          a_cn[iA]
                                      +          rb[iR] * a_cn[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cn[iAp1] * pa[iP1];

               iP1 = iP + xOffsetP;
               rap_ce[iAc] =          a_ce[iA]
                                      +          rb[iR] * a_ce[iAm1] * pb[iP1]
                                      +          ra[iR] * a_ce[iAp1] * pa[iP1];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for 15-point fine grid operator; produces upper triangular
          * part of 15-point coarse grid operator. stencil entries:
          * (above-north, above-east, above-center, above-west,
          * above-south, center-north, and center-east).
          *--------------------------------------------------------------*/

         case 15:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_an,ra,a_cn,pb,a_an,rap_ae,a_ce,a_ae,rap_ac,a_ac,a_cc,rap_aw,a_cw,a_aw,rap_as,a_cs,a_as,rap_cn,rb,pa,a_bn,rap_ce,a_be)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP + zOffsetP + yOffsetP;
               rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                             +          ra[iR] * a_an[iAp1]
                             +                   a_an[iA]   * pb[iP1];

               iP1 = iP + zOffsetP + xOffsetP;
               rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                             +          ra[iR] * a_ae[iAp1]
                             +                   a_ae[iA]   * pb[iP1];

               iP1 = iP + zOffsetP;
               rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                                      +          ra[iR] * a_cc[iAp1] * pb[iP1]
                                      +          ra[iR] * a_ac[iAp1];

               iP1 = iP + zOffsetP - xOffsetP;
               rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                             +          ra[iR] * a_aw[iAp1]
                             +                   a_aw[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP;
               rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                             +          ra[iR] * a_as[iAp1]
                             +                   a_as[iA]   * pb[iP1];

               iP1 = iP + yOffsetP;
               rap_cn[iAc] =          a_cn[iA]
                                      +          rb[iR] * a_cn[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cn[iAp1] * pa[iP1]
                                      +                   a_bn[iA]   * pb[iP1]
                                      +                   a_an[iA]   * pa[iP1]
                                      +          rb[iR] * a_an[iAm1]
                                      +          ra[iR] * a_bn[iAp1];

               iP1 = iP + xOffsetP;
               rap_ce[iAc] =          a_ce[iA]
                                      +          rb[iR] * a_ce[iAm1] * pb[iP1]
                                      +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                      +                   a_be[iA]   * pb[iP1]
                                      +                   a_ae[iA]   * pa[iP1]
                                      +          rb[iR] * a_ae[iAm1]
                                      +          ra[iR] * a_be[iAp1];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;


         /*--------------------------------------------------------------
          * Loop for 19-point fine grid operator; produces upper triangular
          * part of 27-point coarse grid operator. stencil entries:
          * (above-northeast, above-north, above-northwest, above-east,
          * above-center, above-west, above-southeast, above-south,
          * above-southwest, center-northeast, center-north,
          * center-northwest, and center-east).
          *--------------------------------------------------------------*/

         case 19:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_ane,ra,a_cne,pb,rap_an,a_cn,a_an,rap_anw,a_cnw,rap_ae,a_ce,a_ae,rap_ac,a_ac,a_cc,rap_aw,a_cw,a_aw,rap_ase,a_cse,rap_as,a_cs,a_as,rap_asw,a_csw,rap_cne,rb,pa,rap_cn,a_bn,rap_cnw,rap_ce,a_be)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int  iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
               rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP + yOffsetP;
               rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                             +          ra[iR] * a_an[iAp1]
                             +                   a_an[iA]   * pb[iP1];

               iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
               rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP + xOffsetP;
               rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                             +          ra[iR] * a_ae[iAp1]
                             +                   a_ae[iA]   * pb[iP1];

               iP1 = iP + zOffsetP;
               rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                                      +          ra[iR] * a_cc[iAp1] * pb[iP1]
                                      +          ra[iR] * a_ac[iAp1];

               iP1 = iP + zOffsetP - xOffsetP;
               rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                             +          ra[iR] * a_aw[iAp1]
                             +                   a_aw[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
               rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP;
               rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                             +          ra[iR] * a_as[iAp1]
                             +                   a_as[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
               rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1];

               iP1 = iP + yOffsetP + xOffsetP;
               rap_cne[iAc] =         a_cne[iA]
                                      +          rb[iR] * a_cne[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cne[iAp1] * pa[iP1];

               iP1 = iP + yOffsetP;
               rap_cn[iAc] =          a_cn[iA]
                                      +          rb[iR] * a_cn[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cn[iAp1] * pa[iP1]
                                      +                   a_bn[iA]   * pb[iP1]
                                      +                   a_an[iA]   * pa[iP1]
                                      +          rb[iR] * a_an[iAm1]
                                      +          ra[iR] * a_bn[iAp1];

               iP1 = iP + yOffsetP - xOffsetP;
               rap_cnw[iAc] =         a_cnw[iA]
                                      +          rb[iR] * a_cnw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cnw[iAp1] * pa[iP1];

               iP1 = iP + xOffsetP;
               rap_ce[iAc] =          a_ce[iA]
                                      +          rb[iR] * a_ce[iAm1] * pb[iP1]
                                      +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                      +                   a_be[iA]   * pb[iP1]
                                      +                   a_ae[iA]   * pa[iP1]
                                      +          rb[iR] * a_ae[iAm1]
                                      +          ra[iR] * a_be[iAp1];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for 27-point fine grid operator; produces upper triangular
          * part of 27-point coarse grid operator. stencil entries:
          * (above-northeast, above-north, above-northwest, above-east,
          * above-center, above-west, above-southeast, above-south,
          * above-southwest, center-northeast, center-north,
          * center-northwest, and center-east).
          *--------------------------------------------------------------*/

         default:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_ane,ra,a_cne,pb,a_ane,rap_an,a_cn,a_an,rap_anw,a_cnw,a_anw,rap_ae,a_ce,a_ae,rap_ac,a_ac,a_cc,rap_aw,a_cw,a_aw,rap_ase,a_cse,a_ase,rap_as,a_cs,a_as,rap_asw,a_csw,a_asw,rap_cne,rb,pa,a_bne,rap_cn,a_bn,rap_cnw,a_bnw,rap_ce,a_be)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - zOffsetA;
               HYPRE_Int iAp1 = iA + zOffsetA;

               HYPRE_Int iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
               rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1]
                              +           ra[iR] * a_ane[iAp1]
                              +                    a_ane[iA]   * pb[iP1];

               iP1 = iP + zOffsetP + yOffsetP;
               rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                             +          ra[iR] * a_an[iAp1]
                             +                   a_an[iA]   * pb[iP1];

               iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
               rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1]
                              +           ra[iR] * a_anw[iAp1]
                              +                    a_anw[iA]   * pb[iP1];

               iP1 = iP + zOffsetP + xOffsetP;
               rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                             +          ra[iR] * a_ae[iAp1]
                             +                   a_ae[iA]   * pb[iP1];

               iP1 = iP + zOffsetP;
               rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                                      +          ra[iR] * a_cc[iAp1] * pb[iP1]
                                      +          ra[iR] * a_ac[iAp1];

               iP1 = iP + zOffsetP - xOffsetP;
               rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                             +          ra[iR] * a_aw[iAp1]
                             +                   a_aw[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
               rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1]
                              +           ra[iR] * a_ase[iAp1]
                              +                    a_ase[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP;
               rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                             +          ra[iR] * a_as[iAp1]
                             +                   a_as[iA]   * pb[iP1];

               iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
               rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1]
                              +           ra[iR] * a_asw[iAp1]
                              +                    a_asw[iA]   * pb[iP1];


               iP1 = iP + yOffsetP + xOffsetP;
               rap_cne[iAc] =         a_cne[iA]
                                      +          rb[iR] * a_cne[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cne[iAp1] * pa[iP1]
                                      +                   a_bne[iA]   * pb[iP1]
                                      +                   a_ane[iA]   * pa[iP1]
                                      +          rb[iR] * a_ane[iAm1]
                                      +          ra[iR] * a_bne[iAp1];

               iP1 = iP + yOffsetP;
               rap_cn[iAc] =          a_cn[iA]
                                      +          rb[iR] * a_cn[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cn[iAp1] * pa[iP1]
                                      +                   a_bn[iA]   * pb[iP1]
                                      +                   a_an[iA]   * pa[iP1]
                                      +          rb[iR] * a_an[iAm1]
                                      +          ra[iR] * a_bn[iAp1];

               iP1 = iP + yOffsetP - xOffsetP;
               rap_cnw[iAc] =         a_cnw[iA]
                                      +          rb[iR] * a_cnw[iAm1] * pb[iP1]
                                      +          ra[iR] * a_cnw[iAp1] * pa[iP1]
                                      +                   a_bnw[iA]   * pb[iP1]
                                      +                   a_anw[iA]   * pa[iP1]
                                      +          rb[iR] * a_anw[iAm1]
                                      +          ra[iR] * a_bnw[iAp1];

               iP1 = iP + xOffsetP;
               rap_ce[iAc] =          a_ce[iA]
                                      +          rb[iR] * a_ce[iAm1] * pb[iP1]
                                      +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                      +                   a_be[iA]   * pb[iP1]
                                      +                   a_ae[iA]   * pa[iP1]
                                      +          rb[iR] * a_ae[iAm1]
                                      +          ra[iR] * a_be[iAp1];

            }
            hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Collapses stencil in periodic direction on coarsest grid.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMG3RAPPeriodicSym( hypre_StructMatrix *RAP,
                          hypre_Index         cindex,
                          hypre_Index         cstride )

{
   HYPRE_UNUSED_VAR(cindex);
   HYPRE_UNUSED_VAR(cstride);

   hypre_Index             index;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             loop_size;

   HYPRE_Int            ci;

   hypre_Box           *RAP_dbox;

   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_cc, *rap_cw,  *rap_cs;
   HYPRE_Real           *rap_bsw = NULL, *rap_bse = NULL;
   HYPRE_Real           *rap_bnw = NULL, *rap_bne = NULL;
   HYPRE_Real           *rap_csw = NULL, *rap_cse = NULL;

   HYPRE_Int            xOffset;
   HYPRE_Int            yOffset;

   HYPRE_Real           zero = 0.0;

   hypre_StructStencil *stencil;
   HYPRE_Int            stencil_size;

   stencil = hypre_StructMatrixStencil(RAP);
   stencil_size = hypre_StructStencilSize(stencil);

   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   if (hypre_IndexZ(hypre_StructGridPeriodic(cgrid)) == 1)
   {
      hypre_StructMatrixAssemble(RAP);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = hypre_BoxIMin(cgrid_box);

         RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

         hypre_SetIndex3(index, 1, 0, 0);
         xOffset = hypre_BoxOffsetDistance(RAP_dbox, index);
         hypre_SetIndex3(index, 0, 1, 0);
         yOffset = hypre_BoxOffsetDistance(RAP_dbox, index);


         /*-----------------------------------------------------------------
          * Extract pointers for 15-point coarse grid operator:
          *-----------------------------------------------------------------*/

         hypre_SetIndex3(index, 0, 0, -1);
         rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 0, -1);
         rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 0, -1);
         rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, -1, -1);
         rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 1, -1);
         rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 0, 0);
         rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 0, 0);
         rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, -1, 0);
         rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         /*-----------------------------------------------------------------
          * Extract additional pointers for 27-point coarse grid operator:
          *-----------------------------------------------------------------*/

         if (stencil_size == 27)
         {
            hypre_SetIndex3(index, -1, -1, -1);
            rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, -1, -1);
            rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, 1, -1);
            rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, 1, -1);
            rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, -1, 0);
            rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, -1, 0);
            rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);
         }

         /*-----------------------------------------------------------------
          * Collapse 15 point operator.
          *-----------------------------------------------------------------*/

         hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_cc,rap_bc,rap_cw,rap_bw,rap_be,rap_cs,rap_bs,rap_bn)
         hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                             RAP_dbox,  cstart, stridec, iAc);
         {
            HYPRE_Int iAcmx = iAc - xOffset;
            HYPRE_Int iAcmy = iAc - yOffset;

            rap_cc[iAc] += (2.0 * rap_bc[iAc]);
            rap_cw[iAc] += (rap_bw[iAc] + rap_be[iAcmx]);
            rap_cs[iAc] += (rap_bs[iAc] + rap_bn[iAcmy]);
         }
         hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR

#define DEVICE_VAR is_device_ptr(rap_bc,rap_bw,rap_be,rap_bs,rap_bn)
         hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                             RAP_dbox,  cstart, stridec, iAc);
         {
            rap_bc[iAc]  = zero;
            rap_bw[iAc]  = zero;
            rap_be[iAc]  = zero;
            rap_bs[iAc]  = zero;
            rap_bn[iAc]  = zero;
         }
         hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR

         /*-----------------------------------------------------------------
          * Collapse additional entries for 27 point operator.
          *-----------------------------------------------------------------*/

         if (stencil_size == 27)
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_csw,rap_bsw,rap_bne,rap_cse,rap_bse,rap_bnw)
            hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                                RAP_dbox,  cstart, stridec, iAc);
            {
               HYPRE_Int iAcmxmy = iAc - xOffset - yOffset;
               HYPRE_Int iAcpxmy = iAc + xOffset - yOffset;

               rap_csw[iAc] += (rap_bsw[iAc] + rap_bne[iAcmxmy]);

               rap_cse[iAc] += (rap_bse[iAc] + rap_bnw[iAcpxmy]);

            }
            hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR

#define DEVICE_VAR is_device_ptr(rap_bsw,rap_bse,rap_bnw,rap_bne)
            hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                                RAP_dbox,  cstart, stridec, iAc);
            {
               rap_bsw[iAc]  = zero;
               rap_bse[iAc]  = zero;
               rap_bnw[iAc]  = zero;
               rap_bne[iAc]  = zero;
            }
            hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

      } /* end ForBoxI */

   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Collapses stencil in periodic direction on coarsest grid.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMG3RAPPeriodicNoSym( hypre_StructMatrix *RAP,
                            hypre_Index         cindex,
                            hypre_Index         cstride )

{
   HYPRE_UNUSED_VAR(cindex);
   HYPRE_UNUSED_VAR(cstride);

   hypre_Index             index;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             loop_size;

   HYPRE_Int            ci;

   hypre_Box           *RAP_dbox;

   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_cc, *rap_cw, *rap_ce, *rap_cs, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;
   HYPRE_Real           *rap_csw, *rap_cse, *rap_cnw, *rap_cne;
   HYPRE_Real           *rap_asw, *rap_ase, *rap_anw, *rap_ane;

   HYPRE_Real           zero = 0.0;

   hypre_StructStencil *stencil;
   HYPRE_Int            stencil_size;

   stencil = hypre_StructMatrixStencil(RAP);
   stencil_size = hypre_StructStencilSize(stencil);

   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   if (hypre_IndexZ(hypre_StructGridPeriodic(cgrid)) == 1)
   {
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = hypre_BoxIMin(cgrid_box);

         RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

         /*-----------------------------------------------------------------
          * Extract pointers for 15-point coarse grid operator:
          *-----------------------------------------------------------------*/

         hypre_SetIndex3(index, 0, 0, -1);
         rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 0, -1);
         rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 0, -1);
         rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, -1, -1);
         rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 1, -1);
         rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 0, 0);
         rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 0, 0);
         rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 0, 0);
         rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, -1, 0);
         rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 1, 0);
         rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 0, 1);
         rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, -1, 0, 1);
         rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 1, 0, 1);
         rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, -1, 1);
         rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex3(index, 0, 1, 1);
         rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         /*-----------------------------------------------------------------
          * Extract additional pointers for 27-point coarse grid operator:
          *-----------------------------------------------------------------*/

         if (stencil_size == 27)
         {

            hypre_SetIndex3(index, -1, -1, -1);
            rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, -1, -1);
            rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, 1, -1);
            rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, 1, -1);
            rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, -1, 0);
            rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, -1, 0);
            rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, 1, 0);
            rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, 1, 0);
            rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, -1, 1);
            rap_asw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, -1, 1);
            rap_ase = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, -1, 1, 1);
            rap_anw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

            hypre_SetIndex3(index, 1, 1, 1);
            rap_ane = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         }

         /*-----------------------------------------------------------------
          * Collapse 15 point operator.
          *-----------------------------------------------------------------*/

         hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_cc,rap_bc,rap_ac,rap_cw,rap_bw,rap_aw,rap_ce,rap_be,rap_ae,rap_cs,rap_bs,rap_as,rap_cn,rap_bn,rap_an)
         hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                             RAP_dbox,  cstart, stridec, iAc);
         {
            rap_cc[iAc] += (rap_bc[iAc] + rap_ac[iAc]);
            rap_bc[iAc]  = zero;
            rap_ac[iAc]  = zero;

            rap_cw[iAc] += (rap_bw[iAc] + rap_aw[iAc]);
            rap_bw[iAc]  = zero;
            rap_aw[iAc]  = zero;

            rap_ce[iAc] += (rap_be[iAc] + rap_ae[iAc]);
            rap_be[iAc]  = zero;
            rap_ae[iAc]  = zero;

            rap_cs[iAc] += (rap_bs[iAc] + rap_as[iAc]);
            rap_bs[iAc]  = zero;
            rap_as[iAc]  = zero;

            rap_cn[iAc] += (rap_bn[iAc] + rap_an[iAc]);
            rap_bn[iAc]  = zero;
            rap_an[iAc]  = zero;
         }
         hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR

         /*-----------------------------------------------------------------
          * Collapse additional entries for 27 point operator.
          *-----------------------------------------------------------------*/

         if (stencil_size == 27)
         {
            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_csw,rap_bsw,rap_asw,rap_cse,rap_bse,rap_ase,rap_cnw,rap_bnw,rap_anw,rap_cne,rap_bne,rap_ane)
            hypre_BoxLoop1Begin(hypre_StructMatrixNDim(RAP), loop_size,
                                RAP_dbox,  cstart, stridec, iAc);
            {
               rap_csw[iAc] += (rap_bsw[iAc] + rap_asw[iAc]);
               rap_bsw[iAc]  = zero;
               rap_asw[iAc]  = zero;

               rap_cse[iAc] += (rap_bse[iAc] + rap_ase[iAc]);
               rap_bse[iAc]  = zero;
               rap_ase[iAc]  = zero;

               rap_cnw[iAc] += (rap_bnw[iAc] + rap_anw[iAc]);
               rap_bnw[iAc]  = zero;
               rap_anw[iAc]  = zero;

               rap_cne[iAc] += (rap_bne[iAc] + rap_ane[iAc]);
               rap_bne[iAc]  = zero;
               rap_ane[iAc]  = zero;
            }
            hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

      } /* end ForBoxI */

   }

   return hypre_error_flag;
}

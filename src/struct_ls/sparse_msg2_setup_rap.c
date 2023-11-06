/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

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
 * hypre_SparseMSG2CreateRAPOp
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SparseMSG2CreateRAPOp( hypre_StructMatrix *R,
                             hypre_StructMatrix *A,
                             hypre_StructMatrix *P,
                             hypre_StructGrid   *coarse_grid,
                             HYPRE_Int           cdir        )
{
   HYPRE_UNUSED_VAR(R);
   HYPRE_UNUSED_VAR(P);

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
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
      for (j = -1; j < 2; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Storage for 9 elements (c,w,e,n,s,sw,se,nw,se)
             *--------------------------------------------------------------*/
            hypre_SetIndex3(index_temp, i, j, 0);
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
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
      for (j = -1; j < 1; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Store 5 elements in (c,w,s,sw,se)
             *--------------------------------------------------------------*/
            if ( i + j <= 0 )
            {
               hypre_SetIndex3(index_temp, i, j, 0);
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
 * I've written two routines - hypre_SparseMSG2BuildRAPSym to build the
 * lower triangular part of RAP (including the diagonal) and
 * hypre_SparseMSG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the
 * first routine would be called. With full storage both would need to
 * be called.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSG2BuildRAPSym( hypre_StructMatrix *A,
                             hypre_StructMatrix *P,
                             hypre_StructMatrix *R,
                             HYPRE_Int           cdir,
                             hypre_Index         cindex,
                             hypre_Index         cstride,
                             hypre_Index         stridePR,
                             hypre_StructMatrix *RAP      )
{

   hypre_Index           index;
   hypre_Index           index_temp;

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
   hypre_Index           Pstart;
   hypre_Index           loop_size;

   HYPRE_Int             fi, ci;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_csw = NULL, *a_cse = NULL, *a_cnw = NULL;

   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_csw, *rap_cse;

   HYPRE_Int             yOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;

   HYPRE_Int             ierr = 0;

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
      hypre_StructMapCoarseToFine(cstart, cindex, cstride,  fstart);
      hypre_StructMapCoarseToFine(cstart, cindex, stridePR, Pstart);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
      P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
      R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
      RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pa is pointer for weight for f-point above c-point
       * pb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
      MapIndex(index_temp, cdir, index);
      pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
           hypre_BoxOffsetDistance(P_dbox, index);

      /*-----------------------------------------------------------------
       * Extract pointers for restriction operator:
       * ra is pointer for weight for f-point above c-point
       * rb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
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

      hypre_SetIndex3(index_temp, 0, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, -1, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, 1, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
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

      if (fine_stencil_size > 5)
      {
         hypre_SetIndex3(index_temp, -1, -1, 0);
         MapIndex(index_temp, cdir, index);
         a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index_temp, 1, -1, 0);
         MapIndex(index_temp, cdir, index);
         a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index_temp, -1, 1, 0);
         MapIndex(index_temp, cdir, index);
         a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
      }

      /*-----------------------------------------------------------------
       * Extract pointers for coarse grid operator - always 9-point:
       *
       * We build only the lower triangular part (plus diagonal).
       *
       * rap_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, 0, 0);
      MapIndex(index_temp, cdir, index);
      rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, -1, 0, 0);
      MapIndex(index_temp, cdir, index);
      rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, -1, -1, 0);
      MapIndex(index_temp, cdir, index);
      rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, 1, -1, 0);
      MapIndex(index_temp, cdir, index);
      rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets are used in refering to data associated with other points.
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, 1, 0);
      MapIndex(index_temp, cdir, index);
      yOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
      yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
      hypre_SetIndex3(index_temp, 1, 0, 0);
      MapIndex(index_temp, cdir, index);
      xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

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

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_csw,rb,a_cw,pa,rap_cs,a_cc,a_cs,rap_cse,a_ce,rap_cw,pb,ra,rap_cc,a_cn)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                P_dbox, Pstart, stridePR, iP,
                                R_dbox, Pstart, stridePR, iR,
                                A_dbox, fstart, stridef,  iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - yOffsetA;
               HYPRE_Int iAp1 = iA + yOffsetA;

               HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
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
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 9-point fine grid operator; produces a
          * symmetric 9-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (southwest, south, southeast,
          * west, and center).
          *--------------------------------------------------------------*/

         default:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_csw,rb,a_cw,pa,a_csw,rap_cs,a_cc,a_cs,rap_cse,a_ce,a_cse,rap_cw,pb,ra,a_cnw,rap_cc,a_cn)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                P_dbox, Pstart, stridePR, iP,
                                R_dbox, Pstart, stridePR, iR,
                                A_dbox, fstart, stridef,  iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - yOffsetA;
               HYPRE_Int iAp1 = iA + yOffsetA;

               HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
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
#undef DEVICE_VAR

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSG2BuildRAPNoSym( hypre_StructMatrix *A,
                               hypre_StructMatrix *P,
                               hypre_StructMatrix *R,
                               HYPRE_Int           cdir,
                               hypre_Index         cindex,
                               hypre_Index         cstride,
                               hypre_Index         stridePR,
                               hypre_StructMatrix *RAP      )
{

   hypre_Index           index;
   hypre_Index           index_temp;

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
   hypre_Index           Pstart;
   hypre_Index           loop_size;

   HYPRE_Int             fi, ci;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc = NULL, *a_cw = NULL, *a_ce = NULL, *a_cn = NULL;
   HYPRE_Real           *a_cse = NULL, *a_cnw = NULL, *a_cne = NULL;

   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_cnw, *rap_cne;

   HYPRE_Int             yOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;

   HYPRE_Int             ierr = 0;

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
      hypre_StructMapCoarseToFine(cstart, cindex, cstride,  fstart);
      hypre_StructMapCoarseToFine(cstart, cindex, stridePR, Pstart);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
      P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
      R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
      RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pa is pointer for weight for f-point above c-point
       * pb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
      MapIndex(index_temp, cdir, index);
      pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
           hypre_BoxOffsetDistance(P_dbox, index);

      /*-----------------------------------------------------------------
       * Extract pointers for restriction operator:
       * ra is pointer for weight for f-point above c-point
       * rb is pointer for weight for f-point below c-point
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, -1, 0);
      MapIndex(index_temp, cdir, index);
      ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
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

      hypre_SetIndex3(index_temp, 0, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, -1, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, 1, 0, 0);
      MapIndex(index_temp, cdir, index);
      a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
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

      if (fine_stencil_size > 5)
      {
         hypre_SetIndex3(index_temp, 1, -1, 0);
         MapIndex(index_temp, cdir, index);
         a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index_temp, -1, 1, 0);
         MapIndex(index_temp, cdir, index);
         a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex3(index_temp, 1, 1, 0);
         MapIndex(index_temp, cdir, index);
         a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      }

      /*-----------------------------------------------------------------
       * Extract pointers for coarse grid operator - always 9-point:
       *
       * We build only the upper triangular part.
       *
       * rap_ce is pointer for east coefficient (etc.)
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 1, 0, 0);
      MapIndex(index_temp, cdir, index);
      rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, 0, 1, 0);
      MapIndex(index_temp, cdir, index);
      rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, 1, 1, 0);
      MapIndex(index_temp, cdir, index);
      rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      hypre_SetIndex3(index_temp, -1, 1, 0);
      MapIndex(index_temp, cdir, index);
      rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets are used in refering to data associated with other points.
       *-----------------------------------------------------------------*/

      hypre_SetIndex3(index_temp, 0, 1, 0);
      MapIndex(index_temp, cdir, index);
      yOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
      yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
      hypre_SetIndex3(index_temp, 1, 0, 0);
      MapIndex(index_temp, cdir, index);
      xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

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

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_cne,ra,a_ce,pb,rap_cn,a_cc,a_cn,rap_cnw,a_cw,rap_ce,rb,pa)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                P_dbox, Pstart, stridePR, iP,
                                R_dbox, Pstart, stridePR, iR,
                                A_dbox, fstart, stridef,  iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - yOffsetA;
               HYPRE_Int iAp1 = iA + yOffsetA;

               HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
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
#undef DEVICE_VAR

            break;

         /*--------------------------------------------------------------
          * Loop for 9-point fine grid operator; produces upper triangular
          * part of 9-point coarse grid operator - excludes diagonal.
          * stencil entries: (northeast, north, northwest, and east)
          *--------------------------------------------------------------*/

         default:

            hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(rap_cne,ra,a_ce,pb,a_cne,rap_cn,a_cc,a_cn,rap_cnw,a_cw,a_cnw,rap_ce,rb,pa,a_cse)
            hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                                P_dbox, Pstart, stridePR, iP,
                                R_dbox, Pstart, stridePR, iR,
                                A_dbox, fstart, stridef,  iA,
                                RAP_dbox, cstart, stridec, iAc);
            {
               HYPRE_Int iAm1 = iA - yOffsetA;
               HYPRE_Int iAp1 = iA + yOffsetA;

               HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
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
#undef DEVICE_VAR

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}

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

#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * zzz_SMG2NewRAPOp 
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/
 
zzz_StructMatrix *
zzz_SMG2NewRAPOp( zzz_StructMatrix *R,
                  zzz_StructMatrix *A,
                  zzz_StructMatrix *PT )
{
   zzz_StructMatrix    *RAP;

   zzz_StructGrid      *coarse_grid;

   zzz_Index           *RAP_stencil_shape;
   zzz_StructStencil   *RAP_stencil;
   int                  RAP_stencil_size;
   int                  RAP_stencil_dim;
   int                  RAP_num_ghost[] = {1, 1, 1, 1, 0, 0};

   zzz_StructStencil   *A_stencil;
   int                  A_stencil_size;

   int                  j, i;
   int                  stencil_rank;
 
   RAP_stencil_dim = 2;

   coarse_grid = zzz_StructMatrixGrid(R);

   A_stencil = zzz_StructMatrixStencil(A);
   A_stencil_size = zzz_StructStencilSize(A_stencil);
 
/*--------------------------------------------------------------------------
 * Define RAP_stencil
 *--------------------------------------------------------------------------*/

   stencil_rank = 0;

/*--------------------------------------------------------------------------
 * non-symmetric case
 *--------------------------------------------------------------------------*/

   if (!zzz_StructMatrixSymmetric(A))
   {

/*--------------------------------------------------------------------------
 *    5 or 9 point fine grid stencil produces 9 point RAP
 *--------------------------------------------------------------------------*/
      RAP_stencil_size = 9;
      RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
      for (j = -1; j < 2; j++)
      {
          for (i = -1; i < 2; i++)
          {

/*--------------------------------------------------------------------------
 *           Storage for 9 elements (c,w,e,n,s,sw,se,nw,se)
 *--------------------------------------------------------------------------*/
             zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,0);
             stencil_rank++;
          }
      }
   }

/*--------------------------------------------------------------------------
 * symmetric case
 *--------------------------------------------------------------------------*/

   else
   {

/*--------------------------------------------------------------------------
 *    5 or 9 point fine grid stencil produces 9 point RAP
 *    Only store the lower triangular part + diagonal = 5 entries,
 *    lower triangular means the lower triangular part on the matrix
 *    in the standard lexicalgraphic ordering.
 *--------------------------------------------------------------------------*/
      RAP_stencil_size = 5;
      RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
      for (j = -1; j < 1; j++)
      {
          for (i = -1; i < 2; i++)
          {

/*--------------------------------------------------------------------------
 *           Store 5 elements in (c,w,s,sw,se)
 *--------------------------------------------------------------------------*/
             if( i+j <=0 )
             {
                zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,0);
                stencil_rank++;
             }
          }
      }
   }

   RAP_stencil = zzz_NewStructStencil(RAP_stencil_dim, RAP_stencil_size,
                                      RAP_stencil_shape);

   RAP = zzz_NewStructMatrix(zzz_StructMatrixComm(A),
                             coarse_grid, RAP_stencil);

/*--------------------------------------------------------------------------
 * Coarse operator in symmetric iff fine operator is
 *--------------------------------------------------------------------------*/
   zzz_StructMatrixSymmetric(RAP) = zzz_StructMatrixSymmetric(A);

/*--------------------------------------------------------------------------
 * Set number of ghost points - one one each boundary
 *--------------------------------------------------------------------------*/
   zzz_SetStructMatrixNumGhost(RAP, RAP_num_ghost);

   zzz_InitializeStructMatrix(RAP);
 
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
 * I've written a two routines - zzz_SMG2BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * zzz_SMG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 *--------------------------------------------------------------------------*/

int
zzz_SMG2BuildRAPSym( zzz_StructMatrix *A,
                     zzz_StructMatrix *PT,
                     zzz_StructMatrix *R,
                     zzz_StructMatrix *RAP,
                     zzz_Index         cindex,
                     zzz_Index         cstride )

{

   zzz_Index             index_temp;

   zzz_StructStencil    *fine_stencil;
   int                   fine_stencil_size;

   zzz_StructGrid       *cgrid;
   zzz_BoxArray         *cgrid_boxes;
   zzz_Box              *cgrid_box;
   zzz_IndexRef          cstart;
   zzz_Index             stridec;
   zzz_Index             fstart;
   zzz_IndexRef          stridef;
   zzz_Index             loop_size;

   int                  i;
   int                  loopi, loopj, loopk;

   zzz_Box              *A_data_box;
   zzz_Box              *PT_data_box;
   zzz_Box              *R_data_box;
   zzz_Box              *RAP_data_box;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *a_csw, *a_cse, *a_cnw, *a_cne;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  yOffsetA; 
   int                  xOffsetP; 
   int                  yOffsetP; 

   int                  ierr;

   fine_stencil = zzz_StructMatrixStencil(A);
   fine_stencil_size = zzz_StructStencilSize(fine_stencil);

   stridef = cstride;
   zzz_SetIndex(stridec, 1, 1, 1);

   cgrid = zzz_StructMatrixGrid(RAP);
   cgrid_boxes = zzz_StructGridBoxes(cgrid);

   zzz_ForBoxI(i, cgrid_boxes)
   {
      cgrid_box = zzz_BoxArrayBox(cgrid_boxes, i);

      cstart = zzz_BoxIMin(cgrid_box);
      zzz_SMGMapCoarseToFine(cstart, fstart, cindex, cstride) 

      A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
      PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
      R_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(R), i);
      RAP_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(RAP), i);

/*--------------------------------------------------------------------------
 * Extract pointers for interpolation operator:
 * pa is pointer for weight for f-point above c-point 
 * pb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      pa = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      pb = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      ra = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      rb = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for 5-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient
 * a_ce is pointer for east coefficient
 * a_cs is pointer for south coefficient
 * a_cn is pointer for north coefficient
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      a_cc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      a_cw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,1,0,0);
      a_ce = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      a_cs = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,1,0);
      a_cn = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 9-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient
 * a_cse is pointer for southeast coefficient
 * a_cnw is pointer for northwest coefficient
 * a_cne is pointer for northeast coefficient
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 5)
      {
         zzz_SetIndex(index_temp,-1,-1,0);
         a_csw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,0);
         a_cse = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,0);
         a_cnw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,0);
         a_cne = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Extract pointers for coarse grid operator - always 9-point:
 *
 * We build only the lower triangular part (plus diagonal).
 * 
 * rap_cc is pointer for center coefficient (etc.)
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      rap_cc = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      rap_cw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      rap_cs = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,-1,0);
      rap_csw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,1,-1,0);
      rap_cse = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

/*--------------------------------------------------------------------------
 * Define offsets for fine grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP refer to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in refering to data associated with other points. 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      yOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 
      yOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 
      zzz_SetIndex(index_temp,1,0,0);
      xOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 9-point.
 *--------------------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

/*--------------------------------------------------------------------------
 * Loop for symmetric 5-point fine grid operator; produces a symmetric
 * 9-point coarse grid operator. We calculate only the lower triangular
 * (plus diagonal) stencil entries (southwest, south, southeast, west,
 *  and center).
 *--------------------------------------------------------------------------*/

              case 5:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP - yOffsetP - xOffsetP;
                            rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_cs[iAm1]
                                        +          a_cs[iA] * pa[iP1];

                            iP1 = iP - yOffsetP + xOffsetP;
                            rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1];

                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_cn[iAm1]
                                        + ra[iR] * a_cs[iAp1]
                                        +          a_cs[iA] * pb[iP]
                                        +          a_cn[iA] * pa[iP];

                           });


              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 9-point fine grid operator; produces a symmetric
 * 9-point coarse grid operator. We calculate only the lower triangular
 * (plus diagonal) stencil entries (southwest, south, southeast, west,
 *  and center).
 *--------------------------------------------------------------------------*/

              default:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP - yOffsetP - xOffsetP;
                            rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                         + rb[iR] * a_csw[iAm1]
                                         +          a_csw[iA] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_cs[iAm1]
                                        +          a_cs[iA] * pa[iP1];

                            iP1 = iP - yOffsetP + xOffsetP;
                            rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                         + rb[iR] * a_cse[iAm1]
                                         +          a_cse[iA] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1]
                                        + rb[iR] * a_cnw[iAm1]
                                        + ra[iR] * a_csw[iAp1]
                                        +          a_csw[iA] * pb[iP1]
                                        +          a_cnw[iA] * pa[iP1];

                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_cn[iAm1]
                                        + ra[iR] * a_cs[iAp1]
                                        +          a_cs[iA] * pb[iP]
                                        +          a_cn[iA] * pa[iP];

                           });

              break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
zzz_SMG2BuildRAPNoSym( zzz_StructMatrix *A,
                       zzz_StructMatrix *PT,
                       zzz_StructMatrix *R,
                       zzz_StructMatrix *RAP,
                       zzz_Index         cindex,
                       zzz_Index         cstride )

{

   zzz_Index             index_temp;

   zzz_StructStencil    *fine_stencil;
   int                   fine_stencil_size;

   zzz_StructGrid       *cgrid;
   zzz_BoxArray         *cgrid_boxes;
   zzz_Box              *cgrid_box;
   zzz_IndexRef          cstart;
   zzz_Index             stridec;
   zzz_Index             fstart;
   zzz_IndexRef          stridef;
   zzz_Index             loop_size;

   int                  i;
   int                  loopi, loopj, loopk;

   zzz_Box              *A_data_box;
   zzz_Box              *PT_data_box;
   zzz_Box              *R_data_box;
   zzz_Box              *RAP_data_box;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *a_csw, *a_cse, *a_cnw, *a_cne;

   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  yOffsetA;
   int                  xOffsetP;
   int                  yOffsetP;

   int                  ierr;

   fine_stencil = zzz_StructMatrixStencil(A);
   fine_stencil_size = zzz_StructStencilSize(fine_stencil);

   stridef = cstride;
   zzz_SetIndex(stridec, 1, 1, 1);

   cgrid = zzz_StructMatrixGrid(RAP);
   cgrid_boxes = zzz_StructGridBoxes(cgrid);

   zzz_ForBoxI(i, cgrid_boxes)
   {
      cgrid_box = zzz_BoxArrayBox(cgrid_boxes, i);

      cstart = zzz_BoxIMin(cgrid_box);
      zzz_SMGMapCoarseToFine(cstart, fstart, cindex, cstride)

      A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
      PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
      R_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(R), i);
      RAP_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(RAP), i);

/*--------------------------------------------------------------------------
 * Extract pointers for interpolation operator:
 * pa is pointer for weight for f-point above c-point 
 * pb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      pa = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      pb = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      ra = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      rb = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for 5-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient
 * a_ce is pointer for east coefficient
 * a_cs is pointer for south coefficient
 * a_cn is pointer for north coefficient
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      a_cc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      a_cw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,1,0,0);
      a_ce = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,0);
      a_cs = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,1,0);
      a_cn = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 9-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient
 * a_cse is pointer for southeast coefficient
 * a_cnw is pointer for northwest coefficient
 * a_cne is pointer for northeast coefficient
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 5)
      {
         zzz_SetIndex(index_temp,-1,-1,0);
         a_csw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,0);
         a_cse = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,0);
         a_cnw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,0);
         a_cne = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Extract pointers for coarse grid operator - always 9-point:
 *
 * We build only the upper triangular part.
 *
 * rap_ce is pointer for east coefficient (etc.)
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,1,0,0);
      rap_ce = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,1,0);
      rap_cn = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,1,1,0);
      rap_cne = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,1,0);
      rap_cnw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

/*--------------------------------------------------------------------------
 * Define offsets for fine grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP refer to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in refering to data associated with other points. 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,1,0);
      yOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 
      yOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 
      zzz_SetIndex(index_temp,1,0,0);
      xOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 27-point.
 *--------------------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

/*--------------------------------------------------------------------------
 * Loop for 5-point fine grid operator; produces upper triangular
 * part of 9-point coarse grid operator - excludes diagonal.
 * stencil entries: (northeast, north, northwest, and east)
 *--------------------------------------------------------------------------*/

              case 5:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP + yOffsetP + xOffsetP;
                            rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1]
                                        +          a_cn[iA] * pb[iP1];

                            iP1 = iP + yOffsetP - xOffsetP;
                            rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1];

                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for 9-point fine grid operator; produces upper triangular
 * part of 9-point coarse grid operator - excludes diagonal.
 * stencil entries: (northeast, north, northwest, and east)
 *--------------------------------------------------------------------------*/

              default:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP + yOffsetP + xOffsetP;
                            rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                         + ra[iR] * a_cne[iAp1]
                                         +          a_cne[iA] * pb[iP1];

                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1]
                                        +          a_cn[iA] * pb[iP1];

                            iP1 = iP + yOffsetP - xOffsetP;
                            rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                         + ra[iR] * a_cnw[iAp1]
                                         +          a_cnw[iA] * pb[iP1];

                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1]
                                        + rb[iR] * a_cne[iAm1]
                                        + ra[iR] * a_cse[iAp1]
                                        +          a_cse[iA] * pb[iP1]
                                        +          a_cne[iA] * pa[iP1];

                           });

              break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}


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
 * zzz_SMG3NewRAPOp 
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/
 
zzz_StructMatrix *
zzz_SMG3NewRAPOp( zzz_StructMatrix *R,
                  zzz_StructMatrix *A,
                  zzz_StructMatrix *PT )
{
   zzz_StructMatrix    *RAP;

   zzz_StructGrid      *coarse_grid;

   zzz_Index           *RAP_stencil_shape;
   zzz_StructStencil   *RAP_stencil;
   int                  RAP_stencil_size;
   int                  RAP_stencil_dim;
   int                  RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   zzz_StructStencil   *A_stencil;
   int                  A_stencil_size;

   int                  k, j, i;
   int                  stencil_rank;

   RAP_stencil_dim = 3;
 
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
 *    7 or 15 point fine grid stencil produces 15 point RAP
 *--------------------------------------------------------------------------*/
      if( A_stencil_size <= 15)
      {
         RAP_stencil_size = 15;
         RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
         for (k = -1; k < 2; k++)
         {
            for (j = -1; j < 2; j++)
            {
                for (i = -1; i < 2; i++)
                {

/*--------------------------------------------------------------------------
 *                 Storage for c,w,e,n,s elements in each plane
 *--------------------------------------------------------------------------*/
                   if( i*j == 0 )
                   {
                      zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,k);
                      stencil_rank++;
                   }
                }
             }
          }
      }

/*--------------------------------------------------------------------------
 *    19 or 27 point fine grid stencil produces 27 point RAP
 *--------------------------------------------------------------------------*/
      else
      {
         RAP_stencil_size = 27;
         RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
         for (k = -1; k < 2; k++)
         {
            for (j = -1; j < 2; j++)
            {
                for (i = -1; i < 2; i++)
                {

/*--------------------------------------------------------------------------
 *                 Storage for 9 elements (c,w,e,n,s,sw,se,nw,se) in
 *                 each plane
 *--------------------------------------------------------------------------*/
                   zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,k);
                   stencil_rank++;
                }
             }
          }
      }
   }

/*--------------------------------------------------------------------------
 * symmetric case
 *--------------------------------------------------------------------------*/

   else
   {

/*--------------------------------------------------------------------------
 *    7 or 15 point fine grid stencil produces 15 point RAP
 *    Only store the lower triangular part + diagonal = 8 entries,
 *    lower triangular means the lower triangular part on the matrix
 *    in the standard lexicalgraphic ordering.
 *--------------------------------------------------------------------------*/
      if( A_stencil_size <= 15)
      {
         RAP_stencil_size = 8;
         RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
         for (k = -1; k < 1; k++)
         {
            for (j = -1; j < 2; j++)
            {
                for (i = -1; i < 2; i++)
                {

/*--------------------------------------------------------------------------
 *                 Store  5 elements in lower plane (c,w,e,s,n)  
 *                 and 3 elements in same plane (c,w,s)
 *--------------------------------------------------------------------------*/
                   if( i*j == 0 && i+j+k <= 0)
                   {
                      zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,k);
                      stencil_rank++;
                   }
                }
             }
          }
      }

/*--------------------------------------------------------------------------
 *    19 or 27 point fine grid stencil produces 27 point RAP
 *    Only store the lower triangular part + diagonal = 14 entries,
 *    lower triangular means the lower triangular part on the matrix
 *    in the standard lexicalgraphic ordering.
 *--------------------------------------------------------------------------*/
      else
      {
         RAP_stencil_size = 14;
         RAP_stencil_shape = zzz_CTAlloc(zzz_Index, RAP_stencil_size);
         for (k = -1; k < 1; k++)
         {
            for (j = -1; j < 2; j++)
            {
                for (i = -1; i < 2; i++)
                {

/*--------------------------------------------------------------------------
 *                 Store  9 elements in lower plane (c,w,e,s,n,sw,se,nw,ne)  
 *                 and 5 elements in same plane (c,w,s,sw,se)
 *--------------------------------------------------------------------------*/
                   if( k < 0 || (i+j+k <=0 && j < 1) )
                   {
                      zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,j,k);
                      stencil_rank++;
                   }
                }
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
 *  3) 7,15,19 or 27-point fine grid A 
 *
 * I am, however, assuming that the c-to-c interpolation is the identity.
 *
 * I've written a two routines - zzz_SMG3BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * zzz_SMG3BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 *--------------------------------------------------------------------------*/

int
zzz_SMG3BuildRAPSym( zzz_StructMatrix *A,
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
   double               *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   double               *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   double               *a_csw, *a_cse, *a_cnw, *a_cne;
   double               *a_asw, *a_ase, *a_anw, *a_ane;
   double               *a_bsw, *a_bse, *a_bnw, *a_bne;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   double               *rap_csw, *rap_cse;
   double               *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  zOffsetA; 
   int                  xOffsetP; 
   int                  yOffsetP; 
   int                  zOffsetP; 

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

      zzz_SetIndex(index_temp,0,0,1);
      pa = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      pb = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,1);
      ra = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      rb = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);
 
/*--------------------------------------------------------------------------
 * Extract pointers for 7-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient in same plane
 * a_ce is pointer for east coefficient in same plane
 * a_cs is pointer for south coefficient in same plane
 * a_cn is pointer for north coefficient in same plane
 * a_ac is pointer for center coefficient in plane above
 * a_bc is pointer for center coefficient in plane below
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

      zzz_SetIndex(index_temp,0,0,1);
      a_ac = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      a_bc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

/*--------------------------------------------------------------------------
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
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 7)
      {
         zzz_SetIndex(index_temp,-1,0,1);
         a_aw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,0,1);
         a_ae = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,-1,1);
         a_as = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,1,1);
         a_an = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,0,-1);
         a_bw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,0,-1);
         a_be = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,-1,-1);
         a_bs = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,1,-1);
         a_bn = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }
  
/*--------------------------------------------------------------------------
 * Extract additional pointers for 19-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient in same plane
 * a_cse is pointer for southeast coefficient in same plane
 * a_cnw is pointer for northwest coefficient in same plane
 * a_cne is pointer for northeast coefficient in same plane
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 15)
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
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 19)
      {
         zzz_SetIndex(index_temp,-1,-1,1);
         a_asw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,1);
         a_ase = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,1);
         a_anw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,1);
         a_ane = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,-1,-1);
         a_bsw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,-1);
         a_bse = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,-1);
         a_bnw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,-1);
         a_bne = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Extract pointers for 15-point coarse grid operator:
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

      zzz_SetIndex(index_temp,0,0,-1);
      rap_bc = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,-1);
      rap_bw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,1,0,-1);
      rap_be = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,-1);
      rap_bs = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,1,-1);
      rap_bn = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 27-point coarse grid operator:
 *
 * A 27-point coarse grid operator is produced when the fine grid 
 * stencil is 19 or 27 point.
 *
 * We build only the lower triangular part.
 *
 * rap_csw is pointer for southwest coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 15)
      {
         zzz_SetIndex(index_temp,-1,-1,0);
         rap_csw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,0);
         rap_cse = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,-1,-1,-1);
         rap_bsw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,-1);
         rap_bse = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,-1);
         rap_bnw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,1,-1);
         rap_bne = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Define offsets for fine grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP refer to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in refering to data associated with other points. 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,1);
      zOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 
      zOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 
      zzz_SetIndex(index_temp,0,1,0);
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
 * Loop for symmetric 7-point fine grid operator; produces a symmetric
 * 15-point coarse grid operator. We calculate only the lower triangular
 * stencil entries (below-south, below-west, below-center, below-east,
 * below-north, center-south, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              case 7:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP;
                            rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                        + rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_bc[iAm1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP;
                            rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1];
 
                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] =          a_cs[iA]
                                        + rb[iR] * a_cs[iAm1] * pb[iP1]
                                        + ra[iR] * a_cs[iAp1] * pa[iP1];
 
                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1];
 
                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_ac[iAm1]
                                        + ra[iR] * a_bc[iAp1]
                                        +          a_bc[iA] * pb[iP]
                                        +          a_ac[iA] * pa[iP];

                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 15-point fine grid operator; produces a symmetric
 * 15-point coarse grid operator. Again, we calculate only the lower
 * triangular stencil entries (below-south, below-west, below-center,
 * below-east, below-north, center-south, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              case 15:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP;
                            rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                        + rb[iR] * a_bs[iAm1]
                                        +          a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                        + rb[iR] * a_bw[iAm1]
                                        +          a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                        + rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_bc[iAm1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                        + rb[iR] * a_be[iAm1]
                                        +          a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP;
                            rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                        + rb[iR] * a_bn[iAm1]
                                        +          a_bn[iA] * pa[iP1];
 
                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] =          a_cs[iA]
                                        + rb[iR] * a_cs[iAm1] * pb[iP1]
                                        + ra[iR] * a_cs[iAp1] * pa[iP1]
                                        +          a_bs[iA] * pb[iP1]
                                        +          a_as[iA] * pa[iP1]
                                        + rb[iR] * a_as[iAm1]
                                        + ra[iR] * a_bs[iAp1];
 
                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1]
                                        +          a_bw[iA] * pb[iP1]
                                        +          a_aw[iA] * pa[iP1]
                                        + rb[iR] * a_aw[iAm1]
                                        + ra[iR] * a_bw[iAp1];
 
                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_ac[iAm1]
                                        + ra[iR] * a_bc[iAp1]
                                        +          a_bc[iA] * pb[iP]
                                        +          a_ac[iA] * pa[iP];

                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 19-point fine grid operator; produces a symmetric
 * 27-point coarse grid operator. Again, we calculate only the lower
 * triangular stencil entries (below-southwest, below-south,
 * below-southeast, below-west, below-center, below-east,
 * below-northwest, below-north, below-northeast, center-southwest,
 * center-south, center-southeast, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              case 19:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
                            rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP;
                            rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                        + rb[iR] * a_bs[iAm1]
                                        +          a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
                            rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                        + rb[iR] * a_bw[iAm1]
                                        +          a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                        + rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_bc[iAm1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                        + rb[iR] * a_be[iAm1]
                                        +          a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
                            rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP + yOffsetP;
                            rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                        + rb[iR] * a_bn[iAm1]
                                        +          a_bn[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
                            rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1];

                            iP1 = iP - yOffsetP - xOffsetP;
                            rap_csw[iAc] =          a_csw[iA]
                                         + rb[iR] * a_csw[iAm1] * pb[iP1]
                                         + ra[iR] * a_csw[iAp1] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] =          a_cs[iA]
                                        + rb[iR] * a_cs[iAm1] * pb[iP1]
                                        + ra[iR] * a_cs[iAp1] * pa[iP1]
                                        +          a_bs[iA] * pb[iP1]
                                        +          a_as[iA] * pa[iP1]
                                        + rb[iR] * a_as[iAm1]
                                        + ra[iR] * a_bs[iAp1];
 
                            iP1 = iP - yOffsetP + xOffsetP;
                            rap_cse[iAc] =          a_cse[iA]
                                         + rb[iR] * a_cse[iAm1] * pb[iP1]
                                         + ra[iR] * a_cse[iAp1] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1]
                                        +          a_bw[iA] * pb[iP1]
                                        +          a_aw[iA] * pa[iP1]
                                        + rb[iR] * a_aw[iAm1]
                                        + ra[iR] * a_bw[iAp1];
 
                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_ac[iAm1]
                                        + ra[iR] * a_bc[iAp1]
                                        +          a_bc[iA] * pb[iP]
                                        +          a_ac[iA] * pa[iP];

                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 27-point fine grid operator; produces a symmetric
 * 27-point coarse grid operator. Again, we calculate only the lower
 * triangular stencil entries (below-southwest, below-south,
 * below-southeast, below-west, below-center, below-east,
 * below-northwest, below-north, below-northeast, center-southwest,
 * center-south, center-southeast, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              default:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
                            rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1]
                                         + rb[iR] * a_bsw[iAm1]
                                         +          a_bsw[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP;
                            rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                        + rb[iR] * a_bs[iAm1]
                                        +          a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
                            rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1]
                                         + rb[iR] * a_bse[iAm1]
                                         +          a_bse[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                        + rb[iR] * a_bw[iAm1]
                                        +          a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                        + rb[iR] * a_cc[iAm1] * pa[iP1]
                                        + rb[iR] * a_bc[iAm1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                        + rb[iR] * a_be[iAm1]
                                        +          a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
                            rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1]
                                         + rb[iR] * a_bnw[iAm1]
                                         +          a_bnw[iA] * pa[iP1];

                            iP1 = iP - zOffsetP + yOffsetP;
                            rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                        + rb[iR] * a_bn[iAm1]
                                        +          a_bn[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
                            rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1]
                                         + rb[iR] * a_bne[iAm1]
                                         +          a_bne[iA] * pa[iP1];

                            iP1 = iP - yOffsetP - xOffsetP;
                            rap_csw[iAc] =          a_csw[iA]
                                         + rb[iR] * a_csw[iAm1] * pb[iP1]
                                         + ra[iR] * a_csw[iAp1] * pa[iP1]
                                         +          a_bsw[iA] * pb[iP1]
                                         +          a_asw[iA] * pa[iP1]
                                         + rb[iR] * a_asw[iAm1]
                                         + ra[iR] * a_bsw[iAp1];

                            iP1 = iP - yOffsetP;
                            rap_cs[iAc] =          a_cs[iA]
                                        + rb[iR] * a_cs[iAm1] * pb[iP1]
                                        + ra[iR] * a_cs[iAp1] * pa[iP1]
                                        +          a_bs[iA] * pb[iP1]
                                        +          a_as[iA] * pa[iP1]
                                        + rb[iR] * a_as[iAm1]
                                        + ra[iR] * a_bs[iAp1];
 
                            iP1 = iP - yOffsetP + xOffsetP;
                            rap_cse[iAc] =          a_cse[iA]
                                         + rb[iR] * a_cse[iAm1] * pb[iP1]
                                         + ra[iR] * a_cse[iAp1] * pa[iP1]
                                         +          a_bse[iA] * pb[iP1]
                                         +          a_ase[iA] * pa[iP1]
                                         + rb[iR] * a_ase[iAm1]
                                         + ra[iR] * a_bse[iAp1];

                            iP1 = iP - xOffsetP;
                            rap_cw[iAc] =          a_cw[iA]
                                        + rb[iR] * a_cw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cw[iAp1] * pa[iP1]
                                        +          a_bw[iA] * pb[iP1]
                                        +          a_aw[iA] * pa[iP1]
                                        + rb[iR] * a_aw[iAm1]
                                        + ra[iR] * a_bw[iAp1];
 
                            rap_cc[iAc] =          a_cc[iA]
                                        + rb[iR] * a_cc[iAm1] * pb[iP]
                                        + ra[iR] * a_cc[iAp1] * pa[iP]
                                        + rb[iR] * a_ac[iAm1]
                                        + ra[iR] * a_bc[iAp1]
                                        +          a_bc[iA] * pb[iP]
                                        +          a_ac[iA] * pa[iP];

                           });

              break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
zzz_SMG3BuildRAPNoSym( zzz_StructMatrix *A,
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
   double               *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   double               *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   double               *a_csw, *a_cse, *a_cnw, *a_cne;
   double               *a_asw, *a_ase, *a_anw, *a_ane;
   double               *a_bsw, *a_bse, *a_bnw, *a_bne;

   double               *rap_ce, *rap_cn;
   double               *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   double               *rap_cnw, *rap_cne;
   double               *rap_asw, *rap_ase, *rap_anw, *rap_ane;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  zOffsetA;
   int                  xOffsetP;
   int                  yOffsetP;
   int                  zOffsetP;

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

      zzz_SetIndex(index_temp,0,0,1);
      pa = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      pb = zzz_StructMatrixExtractPointerByIndex(PT, i, index_temp);

 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,1);
      ra = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      rb = zzz_StructMatrixExtractPointerByIndex(R, i, index_temp);

 
/*--------------------------------------------------------------------------
 * Extract pointers for 7-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient in same plane
 * a_ce is pointer for east coefficient in same plane
 * a_cs is pointer for south coefficient in same plane
 * a_cn is pointer for north coefficient in same plane
 * a_ac is pointer for center coefficient in plane above
 * a_bc is pointer for center coefficient in plane below
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

      zzz_SetIndex(index_temp,0,0,1);
      a_ac = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,0,0,-1);
      a_bc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);


/*--------------------------------------------------------------------------
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
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 7)
      {
         zzz_SetIndex(index_temp,-1,0,1);
         a_aw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,0,1);
         a_ae = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,-1,1);
         a_as = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,1,1);
         a_an = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,0,-1);
         a_bw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,0,-1);
         a_be = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,-1,-1);
         a_bs = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,0,1,-1);
         a_bn = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }
  
/*--------------------------------------------------------------------------
 * Extract additional pointers for 19-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient in same plane
 * a_cse is pointer for southeast coefficient in same plane
 * a_cnw is pointer for northwest coefficient in same plane
 * a_cne is pointer for northeast coefficient in same plane
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 15)
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
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 19)
      {
         zzz_SetIndex(index_temp,-1,-1,1);
         a_asw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,1);
         a_ase = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,1);
         a_anw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,1);
         a_ane = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,-1,-1);
         a_bsw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,-1);
         a_bse = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,-1);
         a_bnw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

         zzz_SetIndex(index_temp,1,1,-1);
         a_bne = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Extract pointers for 15-point coarse grid operator:
 *
 * We build only the upper triangular part (excluding diagonal).
 * 
 * rap_ce is pointer for east coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,1,0,0);
      rap_ce = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,1,0);
      rap_cn = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,0,1);
      rap_ac = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,1);
      rap_aw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,1,0,1);
      rap_ae = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,-1,1);
      rap_as = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,0,1,1);
      rap_an = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 27-point coarse grid operator:
 *
 * A 27-point coarse grid operator is produced when the fine grid 
 * stencil is 19 or 27 point.
 *
 * We build only the upper triangular part.
 *
 * rap_cnw is pointer for northwest coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

      if(fine_stencil_size > 15)
      {
         zzz_SetIndex(index_temp,-1,1,0);
         rap_cnw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,1,0);
         rap_cne = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,-1,-1,1);
         rap_asw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,-1,1);
         rap_ase = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,-1,1,1);
         rap_anw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

         zzz_SetIndex(index_temp,1,1,1);
         rap_ane = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      }

/*--------------------------------------------------------------------------
 * Define offsets for fine grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP refer to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in refering to data associated with other points. 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,1);
      zOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 
      zOffsetP = zzz_BoxOffsetDistance(PT_data_box,index_temp); 
      zzz_SetIndex(index_temp,0,1,0);
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
 * Loop for 7-point fine grid operator; produces upper triangular part of
 * 15-point coarse grid operator: 
 * stencil entries (above-north, above-east, above-center, above-west,
 * above-south, center-north, and center-east).
 *--------------------------------------------------------------------------*/

              case 7:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP;
                            rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];
 
                            iP1 = iP + zOffsetP; 
                            rap_ac[iAc] =          a_ac[iA] * pb[iP1]
                                        + ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_ac[iAp1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP;
                            rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1];
 
                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] =          a_cn[iA]
                                        + rb[iR] * a_cn[iAm1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1] * pa[iP1];
 
                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1];
 
                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for 15-point fine grid operator; produces upper triangular part of
 * 15-point coarse grid operator: 
 * stencil entries (above-north, above-east, above-center, above-west,
 * above-south, center-north, and center-east).
 *--------------------------------------------------------------------------*/

              case 15:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP;
                            rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                        + ra[iR] * a_an[iAp1]
                                        +          a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                        + ra[iR] * a_ae[iAp1]
                                        +          a_ae[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP; 
                            rap_ac[iAc] =          a_ac[iA] * pb[iP1]
                                        + ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_ac[iAp1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                        + ra[iR] * a_aw[iAp1]
                                        +          a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP;
                            rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                        + ra[iR] * a_as[iAp1]
                                        +          a_as[iA] * pb[iP1];
 
                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] =          a_cn[iA]
                                        + rb[iR] * a_cn[iAm1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1] * pa[iP1]
                                        +          a_bn[iA] * pb[iP1]
                                        +          a_an[iA] * pa[iP1]
                                        + rb[iR] * a_an[iAm1]
                                        + ra[iR] * a_bn[iAp1];
 
                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1]
                                        +          a_be[iA] * pb[iP1]
                                        +          a_ae[iA] * pa[iP1]
                                        + rb[iR] * a_ae[iAm1]
                                        + ra[iR] * a_be[iAp1];
 
                           });

              break;


/*--------------------------------------------------------------------------
 * Loop for 19-point fine grid operator; produces upper triangular part of
 * 27-point coarse grid operator: 
 * stencil entries (above-northeast, above-north, above-northwest,
 * above-east, above-center, above-west, above-southeast, above-south,
 * above-southwest, center-northeast, center-north, center-noerthwest,
 * and center-east).
 *--------------------------------------------------------------------------*/

              case 19:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
                            rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP;
                            rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                        + ra[iR] * a_an[iAp1]
                                        +          a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
                            rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                        + ra[iR] * a_ae[iAp1]
                                        +          a_ae[iA] * pb[iP1];

                            iP1 = iP + zOffsetP; 
                            rap_ac[iAc] =          a_ac[iA] * pb[iP1]
                                        + ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_ac[iAp1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                        + ra[iR] * a_aw[iAp1]
                                        +          a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
                            rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP - yOffsetP;
                            rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                        + ra[iR] * a_as[iAp1]
                                        +          a_as[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
                            rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1];

                            iP1 = iP + yOffsetP + xOffsetP;
                            rap_cne[iAc] =          a_cne[iA]
                                         + rb[iR] * a_cne[iAm1] * pb[iP1]
                                         + ra[iR] * a_cne[iAp1] * pa[iP1];

                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] =          a_cn[iA]
                                        + rb[iR] * a_cn[iAm1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1] * pa[iP1]
                                        +          a_bn[iA] * pb[iP1]
                                        +          a_an[iA] * pa[iP1]
                                        + rb[iR] * a_an[iAm1]
                                        + ra[iR] * a_bn[iAp1];
 
                            iP1 = iP + yOffsetP - xOffsetP;
                            rap_cnw[iAc] =          a_cnw[iA]
                                         + rb[iR] * a_cnw[iAm1] * pb[iP1]
                                         + ra[iR] * a_cnw[iAp1] * pa[iP1];

                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1]
                                        +          a_be[iA] * pb[iP1]
                                        +          a_ae[iA] * pa[iP1]
                                        + rb[iR] * a_ae[iAm1]
                                        + ra[iR] * a_be[iAp1];
 
                           });

              break;

/*--------------------------------------------------------------------------
 * Loop for 27-point fine grid operator; produces upper triangular part of
 * 27-point coarse grid operator: 
 * stencil entries (above-northeast, above-north, above-northwest,
 * above-east, above-center, above-west, above-southeast, above-south,
 * above-southwest, center-northeast, center-north, center-noerthwest,
 * and center-east).
 *--------------------------------------------------------------------------*/

              default:

              zzz_GetBoxSize(cgrid_box, loop_size);
              zzz_BoxLoop4(loopi, loopj, loopk, loop_size,
                           PT_data_box, cstart, stridec, iP,
                           R_data_box, cstart, stridec, iR,
                           A_data_box, fstart, stridef, iA,
                           RAP_data_box, cstart, stridec, iAc,
                           {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
                            rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1]
                                         + ra[iR] * a_ane[iAp1]
                                         +          a_ane[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP;
                            rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                        + ra[iR] * a_an[iAp1]
                                        +          a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
                            rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1]
                                         + ra[iR] * a_anw[iAp1]
                                         +          a_anw[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                        + ra[iR] * a_ae[iAp1]
                                        +          a_ae[iA] * pb[iP1];

                            iP1 = iP + zOffsetP; 
                            rap_ac[iAc] =          a_ac[iA] * pb[iP1]
                                        + ra[iR] * a_cc[iAp1] * pb[iP1]
                                        + ra[iR] * a_ac[iAp1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                        + ra[iR] * a_aw[iAp1]
                                        +          a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
                            rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1]
                                         + ra[iR] * a_ase[iAp1]
                                         +          a_ase[iA] * pb[iP1];

                            iP1 = iP + zOffsetP - yOffsetP;
                            rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                        + ra[iR] * a_as[iAp1]
                                        +          a_as[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
                            rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1]
                                         + ra[iR] * a_asw[iAp1]
                                         +          a_asw[iA] * pb[iP1];


                            iP1 = iP + yOffsetP + xOffsetP;
                            rap_cne[iAc] =          a_cne[iA]
                                         + rb[iR] * a_cne[iAm1] * pb[iP1]
                                         + ra[iR] * a_cne[iAp1] * pa[iP1]
                                         +          a_bne[iA] * pb[iP1]
                                         +          a_ane[iA] * pa[iP1]
                                         + rb[iR] * a_ane[iAm1]
                                         + ra[iR] * a_bne[iAp1];

                            iP1 = iP + yOffsetP;
                            rap_cn[iAc] =          a_cn[iA]
                                        + rb[iR] * a_cn[iAm1] * pb[iP1]
                                        + ra[iR] * a_cn[iAp1] * pa[iP1]
                                        +          a_bn[iA] * pb[iP1]
                                        +          a_an[iA] * pa[iP1]
                                        + rb[iR] * a_an[iAm1]
                                        + ra[iR] * a_bn[iAp1];
 
                            iP1 = iP + yOffsetP - xOffsetP;
                            rap_cnw[iAc] =          a_cnw[iA]
                                         + rb[iR] * a_cnw[iAm1] * pb[iP1]
                                         + ra[iR] * a_cnw[iAp1] * pa[iP1]
                                         +          a_bnw[iA] * pb[iP1]
                                         +          a_anw[iA] * pa[iP1]
                                         + rb[iR] * a_anw[iAm1]
                                         + ra[iR] * a_bnw[iAp1];

                            iP1 = iP + xOffsetP;
                            rap_ce[iAc] =          a_ce[iA]
                                        + rb[iR] * a_ce[iAm1] * pb[iP1]
                                        + ra[iR] * a_ce[iAp1] * pa[iP1]
                                        +          a_be[iA] * pb[iP1]
                                        +          a_ae[iA] * pa[iP1]
                                        + rb[iR] * a_ae[iAm1]
                                        + ra[iR] * a_be[iAp1];
 
                           });

              break;

      } /* end switch statement */

   } /* end ForBoxI */

   return ierr;
}


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

/*--------------------------------------------------------------------------
 * 
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Sketch of routines to build RAP. I'm writing it as general as
 * possible:
 *  1) No assumptions about symmetry of A
 *  2) No assumption that R = transpose(P)
 *  3) General 7,15,19 or 27-point fine grid A 
 *  4) Allowing c to c interpolation/restriction weights (not necessarly = 1)
 *
 * I've written a two routines - zzz_SMG3BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * zzz_SMG3BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 * At the moment I have written a switch statement (based on stencil size)
 * that directs control to the appropriate BoxLoop. All coefficients are
 * calculated in a single BoxLoop. Other possibilities are:
 *
 * i)  a BoxLoop for each coefficient
 * ii) 4 BoxLoops - the first always executed, following ones exectuted
 *                  depending on stencil size. This would cut code length
 *                  and perhaps be easier to maintain (with the switch
 *                  statement, the same code appears multiple time), but
 *                  likely less efficient.
 * 
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Unwritten Functions:
 *  1) ExtractPointer... a function to extract the pointer to
 *  appropriate data. (in parflow it appears to be SubmatrixStencilData)
 *--------------------------------------------------------------------------*/

zzz_SMG3BuildRAPSym(
                   )

{
/*--------------------------------------------------------------------------
 * Extract pointers for interpolation operator:
 * pa is pointer for weight for f-point above c-point 
 * pc is pointer for weight for c-point to c-point 
 * pb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

            pa = ExtractPointerP(P,...);
            pc = ExtractPointerP(P,...);
            pb = ExtractPointerP(P,...);
 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rc is pointer for weight for c-point to c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

            ra = ExtractPointerR(R,...);
            rc = ExtractPointerR(R,...);
            rb = ExtractPointerR(R,...);
 
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

            a_cc = ExtractPointerA(A,...);
            a_cw = ExtractPointerA(A,...);
            a_ce = ExtractPointerA(A,...);
            a_cs = ExtractPointerA(A,...);
            a_cn = ExtractPointerA(A,...);
            a_ac = ExtractPointerA(A,...);
            a_bc = ExtractPointerA(A,...);

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

            if(fine_stencil_num_points > 7)
            {
              a_aw = ExtractPointerA(A,...);
              a_ae = ExtractPointerA(A,...);
              a_as = ExtractPointerA(A,...);
              a_an = ExtractPointerA(A,...);
              a_bw = ExtractPointerA(A,...);
              a_be = ExtractPointerA(A,...);
              a_bs = ExtractPointerA(A,...);
              a_bn = ExtractPointerA(A,...);
            }
  
/*--------------------------------------------------------------------------
 * Extract additional pointers for 19-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient in same plane
 * a_cse is pointer for southeast coefficient in same plane
 * a_cnw is pointer for northwest coefficient in same plane
 * a_cne is pointer for northeast coefficient in same plane
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 15)
            {
              a_csw = ExtractPointerA(A,...);
              a_cse = ExtractPointerA(A,...);
              a_cnw = ExtractPointerA(A,...);
              a_cne = ExtractPointerA(A,...);
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

            if(fine_stencil_num_points > 19)
            {
              a_asw = ExtractPointerA(A,...);
              a_ase = ExtractPointerA(A,...);
              a_anw = ExtractPointerA(A,...);
              a_ane = ExtractPointerA(A,...);
              a_bsw = ExtractPointerA(A,...);
              a_bse = ExtractPointerA(A,...);
              a_bnw = ExtractPointerA(A,...);
              a_bne = ExtractPointerA(A,...);
            }

/*--------------------------------------------------------------------------
 * Extract pointers for 15-point coarse grid operator:
 *
 * We build only the lower triangular part (plus diagonal).
 * 
 * ac_cc is pointer for center coefficient (etc.)
 *--------------------------------------------------------------------------*/

            ac_cc = ExtractPointerAc(Ac,...);
            ac_cw = ExtractPointerAc(Ac,...);
            ac_cs = ExtractPointerAc(Ac,...);
            ac_bc = ExtractPointerAc(Ac,...);
            ac_bw = ExtractPointerAc(Ac,...);
            ac_be = ExtractPointerAc(Ac,...);
            ac_bs = ExtractPointerAc(Ac,...);
            ac_bn = ExtractPointerAc(Ac,...);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 27-point coarse grid operator:
 *
 * A 27-point coarse grid operator is produced when the fine grid 
 * stencil is 19 or 27 point.
 *
 * We build only the lower triangular part.
 *
 * ac_csw is pointer for southwest coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 15)
            {
              ac_csw = ExtractPointerAc(Ac,...);
              ac_cse = ExtractPointerAc(Ac,...);
              ac_bsw = ExtractPointerAc(Ac,...);
              ac_bse = ExtractPointerAc(Ac,...);
              ac_bnw = ExtractPointerAc(Ac,...);
              ac_bne = ExtractPointerAc(Ac,...);
            }

/*--------------------------------------------------------------------------
 * Define pointers and offsets for fine/coarse grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP point to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in pointer arithmetic to point to data associated
 * with other points. 
 *--------------------------------------------------------------------------*/

            iA = ;
            iAc = ;
            iP = ;

            zOffsetA = ; 
            xOffsetP = ;
            yOffsetP = ;
            zOffsetP = ;

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 27-point.
 *--------------------------------------------------------------------------*/

            switch (fine_stencil_num_points)
            {

/*--------------------------------------------------------------------------
 * Loop for symmetric 7-point fine grid operator; produces a symmetric
 * 15-point coarse grid operator. We calculate only the lower triangular
 * stencil entries (below-south, below-west, below-center, below-east,
 * below-north, center-south, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              case 7:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP;
                            ac_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            ac_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            ac_bc[iAc] = rc[iR] * a_bc[iA] * pa[iP1]
                                       + rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_bc[iAm1] * pc[iP1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            ac_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP;
                            ac_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1];
 
                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rc[iR] * a_cs[iA] * pc[iP1]
                                       + rb[iR] * a_cs[iAm1] * pb[iP1]
                                       + ra[iR] * a_cs[iAp1] * pa[iP1];
 
                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_ac[iAm1] * pc[iP]
                                       + ra[iR] * a_bc[iAp1] * pc[iP]
                                       + rc[iR] * a_bc[iA] * pb[iP]
                                       + rc[iR] * a_ac[iA] * pa[iP];

                         });

              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 15-point fine grid operator; produces a symmetric
 * 15-point coarse grid operator. Again, we calculate only the lower
 * triangular stencil entries (below-south, below-west, below-center,
 * below-east, below-north, center-south, center-west, and center-center).
 *--------------------------------------------------------------------------*/

              case 15:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP;
                            ac_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                       + rb[iR] * a_bs[iAm1] * pc[iP1]
                                       + rc[iR] * a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            ac_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                       + rb[iR] * a_bw[iAm1] * pc[iP1]
                                       + rc[iR] * a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            ac_bc[iAc] = rc[iR] * a_bc[iA] * pa[iP1]
                                       + rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_bc[iAm1] * pc[iP1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            ac_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                       + rb[iR] * a_be[iAm1] * pc[iP1]
                                       + rc[iR] * a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP;
                            ac_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                       + rb[iR] * a_bn[iAm1] * pc[iP1]
                                       + rc[iR] * a_bn[iA] * pa[iP1];
 
                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rc[iR] * a_cs[iA] * pc[iP1]
                                       + rb[iR] * a_cs[iAm1] * pb[iP1]
                                       + ra[iR] * a_cs[iAp1] * pa[iP1]
                                       + rc[iR] * a_bs[iA] * pb[iP1]
                                       + rc[iR] * a_as[iA] * pa[iP1]
                                       + rb[iR] * a_as[iAm1] * pc[iP1]
                                       + ra[iR] * a_bs[iAp1] * pc[iP1];
 
                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1]
                                       + rc[iR] * a_bw[iA] * pb[iP1]
                                       + rc[iR] * a_aw[iA] * pa[iP1]
                                       + rb[iR] * a_aw[iAm1] * pc[iP1]
                                       + ra[iR] * a_bw[iAp1] * pc[iP1];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_ac[iAm1] * pc[iP]
                                       + ra[iR] * a_bc[iAp1] * pc[iP]
                                       + rc[iR] * a_bc[iA] * pb[iP]
                                       + rc[iR] * a_ac[iA] * pa[iP];

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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
                            ac_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP;
                            ac_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                       + rb[iR] * a_bs[iAm1] * pc[iP1]
                                       + rc[iR] * a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
                            ac_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            ac_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                       + rb[iR] * a_bw[iAm1] * pc[iP1]
                                       + rc[iR] * a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            ac_bc[iAc] = rc[iR] * a_bc[iA] * pa[iP1]
                                       + rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_bc[iAm1] * pc[iP1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            ac_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                       + rb[iR] * a_be[iAm1] * pc[iP1]
                                       + rc[iR] * a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
                            ac_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1];

                            iP1 = iP - zOffsetP + yOffsetP;
                            ac_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                       + rb[iR] * a_bn[iAm1] * pc[iP1]
                                       + rc[iR] * a_bn[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
                            ac_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1];

                            iP1 = iP - yOffsetP - xOffset;
                            ac_csw[iAc] = rc[iR] * a_csw[iA] * pc[iP1]
                                        + rb[iR] * a_csw[iAm1] * pb[iP1]
                                        + ra[iR] * a_csw[iAp1] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rc[iR] * a_cs[iA] * pc[iP1]
                                       + rb[iR] * a_cs[iAm1] * pb[iP1]
                                       + ra[iR] * a_cs[iAp1] * pa[iP1]
                                       + rc[iR] * a_bs[iA] * pb[iP1]
                                       + rc[iR] * a_as[iA] * pa[iP1]
                                       + rb[iR] * a_as[iAm1] * pc[iP1]
                                       + ra[iR] * a_bs[iAp1] * pc[iP1];
 
                            iP1 = iP - yOffsetP + xOffset;
                            ac_cse[iAc] = rc[iR] * a_cse[iA] * pc[iP1]
                                        + rb[iR] * a_cse[iAm1] * pb[iP1]
                                        + ra[iR] * a_cse[iAp1] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1]
                                       + rc[iR] * a_bw[iA] * pb[iP1]
                                       + rc[iR] * a_aw[iA] * pa[iP1]
                                       + rb[iR] * a_aw[iAm1] * pc[iP1]
                                       + ra[iR] * a_bw[iAp1] * pc[iP1];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_ac[iAm1] * pc[iP]
                                       + ra[iR] * a_bc[iAp1] * pc[iP]
                                       + rc[iR] * a_bc[iA] * pb[iP]
                                       + rc[iR] * a_ac[iA] * pa[iP];

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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
                            ac_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1]
                                        + rb[iR] * a_bsw[iAm1] * pc[iP1]
                                        + rc[iR] * a_bsw[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP;
                            ac_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                                       + rb[iR] * a_bs[iAm1] * pc[iP1]
                                       + rc[iR] * a_bs[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
                            ac_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1]
                                        + rb[iR] * a_bse[iAm1] * pc[iP1]
                                        + rc[iR] * a_bse[iA] * pa[iP1];

                            iP1 = iP - zOffsetP - xOffsetP;
                            ac_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                                       + rb[iR] * a_bw[iAm1] * pc[iP1]
                                       + rc[iR] * a_bw[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP; 
                            ac_bc[iAc] = rc[iR] * a_bc[iA] * pa[iP1]
                                       + rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_bc[iAm1] * pc[iP1];
 
                            iP1 = iP - zOffsetP + xOffsetP;
                            ac_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                                       + rb[iR] * a_be[iAm1] * pc[iP1]
                                       + rc[iR] * a_be[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
                            ac_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1]
                                        + rb[iR] * a_bnw[iAm1] * pc[iP1]
                                        + rc[iR] * a_bnw[iA] * pa[iP1];

                            iP1 = iP - zOffsetP + yOffsetP;
                            ac_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                                       + rb[iR] * a_bn[iAm1] * pc[iP1]
                                       + rc[iR] * a_bn[iA] * pa[iP1];
 
                            iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
                            ac_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1]
                                        + rb[iR] * a_bne[iAm1] * pc[iP1]
                                        + rc[iR] * a_bne[iA] * pa[iP1];

                            iP1 = iP - yOffsetP - xOffset;
                            ac_csw[iAc] = rc[iR] * a_csw[iA] * pc[iP1]
                                        + rb[iR] * a_csw[iAm1] * pb[iP1]
                                        + ra[iR] * a_csw[iAp1] * pa[iP1]
                                        + rc[iR] * a_bsw[iA] * pb[iP1]
                                        + rc[iR] * a_asw[iA] * pa[iP1]
                                        + rb[iR] * a_asw[iAm1] * pc[iP1]
                                        + ra[iR] * a_bsw[iAp1] * pc[iP1];

                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rc[iR] * a_cs[iA] * pc[iP1]
                                       + rb[iR] * a_cs[iAm1] * pb[iP1]
                                       + ra[iR] * a_cs[iAp1] * pa[iP1]
                                       + rc[iR] * a_bs[iA] * pb[iP1]
                                       + rc[iR] * a_as[iA] * pa[iP1]
                                       + rb[iR] * a_as[iAm1] * pc[iP1]
                                       + ra[iR] * a_bs[iAp1] * pc[iP1];
 
                            iP1 = iP - yOffsetP + xOffset;
                            ac_cse[iAc] = rc[iR] * a_cse[iA] * pc[iP1]
                                        + rb[iR] * a_cse[iAm1] * pb[iP1]
                                        + ra[iR] * a_cse[iAp1] * pa[iP1]
                                        + rc[iR] * a_bse[iA] * pb[iP1]
                                        + rc[iR] * a_ase[iA] * pa[iP1]
                                        + rb[iR] * a_ase[iAm1] * pc[iP1]
                                        + ra[iR] * a_bse[iAp1] * pc[iP1];

                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1]
                                       + rc[iR] * a_bw[iA] * pb[iP1]
                                       + rc[iR] * a_aw[iA] * pa[iP1]
                                       + rb[iR] * a_aw[iAm1] * pc[iP1]
                                       + ra[iR] * a_bw[iAp1] * pc[iP1];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_ac[iAm1] * pc[iP]
                                       + ra[iR] * a_bc[iAp1] * pc[iP]
                                       + rc[iR] * a_bc[iA] * pb[iP]
                                       + rc[iR] * a_ac[iA] * pa[iP];

                         });

              break;

            }
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

zzz_SMG3BuildRAPNoSym(
                   )

{
/*--------------------------------------------------------------------------
 * Extract pointers for interpolation operator:
 * pa is pointer for weight for f-point above c-point 
 * pc is pointer for weight for c-point to c-point 
 * pb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

            pa = ExtractPointerP(P,...);
            pc = ExtractPointerP(P,...);
            pb = ExtractPointerP(P,...);
 
/*--------------------------------------------------------------------------
 * Extract pointers for restriction operator:
 * ra is pointer for weight for f-point above c-point 
 * rc is pointer for weight for c-point to c-point 
 * rb is pointer for weight for f-point below c-point 
 *--------------------------------------------------------------------------*/

            ra = ExtractPointerR(R,...);
            rc = ExtractPointerR(R,...);
            rb = ExtractPointerR(R,...);
 
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

            a_cc = ExtractPointerA(A,...);
            a_cw = ExtractPointerA(A,...);
            a_ce = ExtractPointerA(A,...);
            a_cs = ExtractPointerA(A,...);
            a_cn = ExtractPointerA(A,...);
            a_ac = ExtractPointerA(A,...);
            a_bc = ExtractPointerA(A,...);

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

            if(fine_stencil_num_points > 7)
            {
              a_aw = ExtractPointerA(A,...);
              a_ae = ExtractPointerA(A,...);
              a_as = ExtractPointerA(A,...);
              a_an = ExtractPointerA(A,...);
              a_bw = ExtractPointerA(A,...);
              a_be = ExtractPointerA(A,...);
              a_bs = ExtractPointerA(A,...);
              a_bn = ExtractPointerA(A,...);
            }
  
/*--------------------------------------------------------------------------
 * Extract additional pointers for 19-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient in same plane
 * a_cse is pointer for southeast coefficient in same plane
 * a_cnw is pointer for northwest coefficient in same plane
 * a_cne is pointer for northeast coefficient in same plane
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 15)
            {
              a_csw = ExtractPointerA(A,...);
              a_cse = ExtractPointerA(A,...);
              a_cnw = ExtractPointerA(A,...);
              a_cne = ExtractPointerA(A,...);
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

            if(fine_stencil_num_points > 19)
            {
              a_asw = ExtractPointerA(A,...);
              a_ase = ExtractPointerA(A,...);
              a_anw = ExtractPointerA(A,...);
              a_ane = ExtractPointerA(A,...);
              a_bsw = ExtractPointerA(A,...);
              a_bse = ExtractPointerA(A,...);
              a_bnw = ExtractPointerA(A,...);
              a_bne = ExtractPointerA(A,...);
            }

/*--------------------------------------------------------------------------
 * Extract pointers for 15-point coarse grid operator:
 *
 * We build only the upper triangular part (excluding diagonal).
 * 
 * ac_ce is pointer for east coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

            ac_ce = ExtractPointerAc(Ac,...);
            ac_cn = ExtractPointerAc(Ac,...);
            ac_ac = ExtractPointerAc(Ac,...);
            ac_aw = ExtractPointerAc(Ac,...);
            ac_ae = ExtractPointerAc(Ac,...);
            ac_as = ExtractPointerAc(Ac,...);
            ac_an = ExtractPointerAc(Ac,...);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 27-point coarse grid operator:
 *
 * A 27-point coarse grid operator is produced when the fine grid 
 * stencil is 19 or 27 point.
 *
 * We build only the upper triangular part.
 *
 * ac_cnw is pointer for northwest coefficient in same plane (etc.)
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 15)
            {
              ac_cnw = ExtractPointerAc(Ac,...);
              ac_cne = ExtractPointerAc(Ac,...);
              ac_asw = ExtractPointerAc(Ac,...);
              ac_ase = ExtractPointerAc(Ac,...);
              ac_anw = ExtractPointerAc(Ac,...);
              ac_ane = ExtractPointerAc(Ac,...);
            }

/*--------------------------------------------------------------------------
 * Define pointers and offsets for fine/coarse grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP point to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in pointer arithmetic to point to data associated
 * with other points. 
 *--------------------------------------------------------------------------*/

            iA = ;
            iAc = ;
            iP = ;

            zOffsetA = ; 
            xOffsetP = ;
            yOffsetP = ;
            zOffsetP = ;

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 27-point.
 *--------------------------------------------------------------------------*/

            switch (fine_stencil_num_points)
            {

/*--------------------------------------------------------------------------
 * Loop for 7-point fine grid operator; produces upper triangular part of
 * 15-point coarse grid operator: 
 * stencil entries (above-north, above-east, above-center, above-west,
 * above-south, center-north, and center-east).
 *--------------------------------------------------------------------------*/

              case 7:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP;
                            ac_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            ac_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];
 
                            iP1 = iP + zOffsetP; 
                            ac_ac[iAc] = rc[iR] * a_ac[iA] * pb[iP1]
                                       + ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_ac[iAp1] * pc[iP1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            ac_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP;
                            ac_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1];
 
                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = rc[iR] * a_cn[iA] * pc[iP1]
                                       + rb[iR] * a_cn[iAm1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pa[iP1];
 
                            iP1 = iP + xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP;
                            ac_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                       + ra[iR] * a_an[iAp1] * pc[iP1]
                                       + rc[iR] * a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            ac_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                       + ra[iR] * a_ae[iAp1] * pc[iP1]
                                       + rc[iR] * a_ae[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP; 
                            ac_ac[iAc] = rc[iR] * a_ac[iA] * pb[iP1]
                                       + ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_ac[iAp1] * pc[iP1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            ac_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                       + ra[iR] * a_aw[iAp1] * pc[iP1]
                                       + rc[iR] * a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP;
                            ac_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                       + ra[iR] * a_as[iAp1] * pc[iP1]
                                       + rc[iR] * a_as[iA] * pb[iP1];
 
                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = rc[iR] * a_cn[iA] * pc[iP1]
                                       + rb[iR] * a_cn[iAm1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pa[iP1]
                                       + rc[iR] * a_bn[iA] * pb[iP1]
                                       + rc[iR] * a_an[iA] * pa[iP1]
                                       + rb[iR] * a_an[iAm1] * pc[iP1]
                                       + ra[iR] * a_bn[iAp1] * pc[iP1];
 
                            iP1 = iP + xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
                                       + rb[iR] * a_ce[iAm1] * pb[iP1]
                                       + ra[iR] * a_ce[iAp1] * pa[iP1]
                                       + rc[iR] * a_be[iA] * pb[iP1]
                                       + rc[iR] * a_ae[iA] * pa[iP1]
                                       + rb[iR] * a_ae[iAm1] * pc[iP1]
                                       + ra[iR] * a_be[iAp1] * pc[iP1];
 
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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
                            ac_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP;
                            ac_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                       + ra[iR] * a_an[iAp1] * pc[iP1]
                                       + rc[iR] * a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
                            ac_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            ac_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                       + ra[iR] * a_ae[iAp1] * pc[iP1]
                                       + rc[iR] * a_ae[iA] * pb[iP1];

                            iP1 = iP + zOffsetP; 
                            ac_ac[iAc] = rc[iR] * a_ac[iA] * pb[iP1]
                                       + ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_ac[iAp1] * pc[iP1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            ac_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                       + ra[iR] * a_aw[iAp1] * pc[iP1]
                                       + rc[iR] * a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
                            ac_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1];

                            iP1 = iP + zOffsetP - yOffsetP;
                            ac_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                       + ra[iR] * a_as[iAp1] * pc[iP1]
                                       + rc[iR] * a_as[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
                            ac_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1];

                            iP1 = iP + yOffsetP + xOffset;
                            ac_cne[iAc] = rc[iR] * a_cne[iA] * pc[iP1]
                                        + rb[iR] * a_cne[iAm1] * pb[iP1]
                                        + ra[iR] * a_cne[iAp1] * pa[iP1];

                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = rc[iR] * a_cn[iA] * pc[iP1]
                                       + rb[iR] * a_cn[iAm1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pa[iP1]
                                       + rc[iR] * a_bn[iA] * pb[iP1]
                                       + rc[iR] * a_an[iA] * pa[iP1]
                                       + rb[iR] * a_an[iAm1] * pc[iP1]
                                       + ra[iR] * a_bn[iAp1] * pc[iP1];
 
                            iP1 = iP + yOffsetP - xOffset;
                            ac_cnw[iAc] = rc[iR] * a_cnw[iA] * pc[iP1]
                                        + rb[iR] * a_cnw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cnw[iAp1] * pa[iP1];

                            iP1 = iP + xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
                                       + rb[iR] * a_ce[iAm1] * pb[iP1]
                                       + ra[iR] * a_ce[iAp1] * pa[iP1]
                                       + rc[iR] * a_be[iA] * pb[iP1]
                                       + rc[iR] * a_ae[iA] * pa[iP1]
                                       + rb[iR] * a_ae[iAm1] * pc[iP1]
                                       + ra[iR] * a_be[iAp1] * pc[iP1];
 
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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - zOffsetA;
                            iAp1 = iA + zOffsetA;

                            iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
                            ac_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1]
                                        + ra[iR] * a_ane[iAp1] * pc[iP1]
                                        + rc[iR] * a_ane[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP;
                            ac_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                                       + ra[iR] * a_an[iAp1] * pc[iP1]
                                       + rc[iR] * a_an[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
                            ac_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1]
                                        + ra[iR] * a_anw[iAp1] * pc[iP1]
                                        + rc[iR] * a_anw[iA] * pb[iP1];

                            iP1 = iP + zOffsetP + xOffsetP;
                            ac_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                                       + ra[iR] * a_ae[iAp1] * pc[iP1]
                                       + rc[iR] * a_ae[iA] * pb[iP1];

                            iP1 = iP + zOffsetP; 
                            ac_ac[iAc] = rc[iR] * a_ac[iA] * pb[iP1]
                                       + ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_ac[iAp1] * pc[iP1];
 
                            iP1 = iP + zOffsetP - xOffsetP;
                            ac_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                                       + ra[iR] * a_aw[iAp1] * pc[iP1]
                                       + rc[iR] * a_aw[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
                            ac_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1]
                                        + ra[iR] * a_ase[iAp1] * pc[iP1]
                                        + rc[iR] * a_ase[iA] * pb[iP1];

                            iP1 = iP + zOffsetP - yOffsetP;
                            ac_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                                       + ra[iR] * a_as[iAp1] * pc[iP1]
                                       + rc[iR] * a_as[iA] * pb[iP1];
 
                            iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
                            ac_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1]
                                        + ra[iR] * a_asw[iAp1] * pc[iP1]
                                        + rc[iR] * a_asw[iA] * pb[iP1];


                            iP1 = iP + yOffsetP + xOffset;
                            ac_cne[iAc] = rc[iR] * a_cne[iA] * pc[iP1]
                                        + rb[iR] * a_cne[iAm1] * pb[iP1]
                                        + ra[iR] * a_cne[iAp1] * pa[iP1]
                                        + rc[iR] * a_bne[iA] * pb[iP1]
                                        + rc[iR] * a_ane[iA] * pa[iP1]
                                        + rb[iR] * a_ane[iAm1] * pc[iP1]
                                        + ra[iR] * a_bne[iAp1] * pc[iP1];

                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = rc[iR] * a_cn[iA] * pc[iP1]
                                       + rb[iR] * a_cn[iAm1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pa[iP1]
                                       + rc[iR] * a_bn[iA] * pb[iP1]
                                       + rc[iR] * a_an[iA] * pa[iP1]
                                       + rb[iR] * a_an[iAm1] * pc[iP1]
                                       + ra[iR] * a_bn[iAp1] * pc[iP1];
 
                            iP1 = iP + yOffsetP - xOffset;
                            ac_cnw[iAc] = rc[iR] * a_cnw[iA] * pc[iP1]
                                        + rb[iR] * a_cnw[iAm1] * pb[iP1]
                                        + ra[iR] * a_cnw[iAp1] * pa[iP1]
                                        + rc[iR] * a_bnw[iA] * pb[iP1]
                                        + rc[iR] * a_anw[iA] * pa[iP1]
                                        + rb[iR] * a_anw[iAm1] * pc[iP1]
                                        + ra[iR] * a_bnw[iAp1] * pc[iP1];

                            iP1 = iP + xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
                                       + rb[iR] * a_ce[iAm1] * pb[iP1]
                                       + ra[iR] * a_ce[iAp1] * pa[iP1]
                                       + rc[iR] * a_be[iA] * pb[iP1]
                                       + rc[iR] * a_ae[iA] * pa[iP1]
                                       + rb[iR] * a_ae[iAm1] * pc[iP1]
                                       + ra[iR] * a_be[iAp1] * pc[iP1];
 
                         });

              break;

            }
}


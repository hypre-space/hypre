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
 *  3) General 5 or 9-point fine grid A 
 *  4) Allowing c to c interpolation/restriction weights (not necessarly = 1)
 *
 * I've written a two routines - zzz_SMG2BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * zzz_SMG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 * At the moment I have written a switch statement (based on stencil size)
 * that directs control to the appropriate BoxLoop. All coefficients are
 * calculated in a single BoxLoop. Other possibilities are:
 *
 * i)  a BoxLoop for each coefficient
 * ii) 2 BoxLoops - the first always executed, following ones exectuted
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

zzz_SMG2BuildRAPSym(
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
 * Extract pointers for 5-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient
 * a_ce is pointer for east coefficient
 * a_cs is pointer for south coefficient
 * a_cn is pointer for north coefficient
 *--------------------------------------------------------------------------*/

            a_cc = ExtractPointerA(A,...);
            a_cw = ExtractPointerA(A,...);
            a_ce = ExtractPointerA(A,...);
            a_cs = ExtractPointerA(A,...);
            a_cn = ExtractPointerA(A,...);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 9-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient
 * a_cse is pointer for southeast coefficient
 * a_cnw is pointer for northwest coefficient
 * a_cne is pointer for northeast coefficient
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 5)
            {
              a_csw = ExtractPointerA(A,...);
              a_cse = ExtractPointerA(A,...);
              a_cnw = ExtractPointerA(A,...);
              a_cne = ExtractPointerA(A,...);
            }

/*--------------------------------------------------------------------------
 * Extract pointers for coarse grid operator - always 9-point:
 *
 * We build only the lower triangular part (plus diagonal).
 * 
 * ac_cc is pointer for center coefficient (etc.)
 *--------------------------------------------------------------------------*/

            ac_cc = ExtractPointerAc(Ac,...);
            ac_cw = ExtractPointerAc(Ac,...);
            ac_cs = ExtractPointerAc(Ac,...);
            ac_csw = ExtractPointerAc(Ac,...);
            ac_cse = ExtractPointerAc(Ac,...);

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

            yOffsetA = ; 
            xOffsetP = ;
            yOffsetP = ;

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 9-point.
 *--------------------------------------------------------------------------*/

            switch (fine_stencil_num_points)
            {

/*--------------------------------------------------------------------------
 * Loop for symmetric 5-point fine grid operator; produces a symmetric
 * 9-point coarse grid operator. We calculate only the lower triangular
 * (plus diagonal) stencil entries (southwest, south, southeast, west,
 *  and center).
 *--------------------------------------------------------------------------*/

              case 5:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP - yOffsetP - xOffset;
                            ac_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_cs[iAm1] * pc[iP1]
                                       + rc[iR] * a_cs[iA] * pa[iP1];
 
                            iP1 = iP - yOffsetP + xOffset;
                            ac_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_cn[iAm1] * pc[iP]
                                       + ra[iR] * a_cs[iAp1] * pc[iP]
                                       + rc[iR] * a_cs[iA] * pb[iP]
                                       + rc[iR] * a_cn[iA] * pa[iP];

                         });

              break;

/*--------------------------------------------------------------------------
 * Loop for symmetric 9-point fine grid operator; produces a symmetric
 * 9-point coarse grid operator. We calculate only the lower triangular
 * (plus diagonal) stencil entries (southwest, south, southeast, west,
 *  and center).
 *--------------------------------------------------------------------------*/


              default:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP - yOffsetP - xOffset;
                            ac_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];
                                        + rb[iR] * a_csw[iAm1] * pc[iP1]
                                        + rc[iR] * a_csw[iA] * pa[iP1];

                            iP1 = iP - yOffsetP;
                            ac_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                                       + rb[iR] * a_cs[iAm1] * pc[iP1]
                                       + rc[iR] * a_cs[iA] * pa[iP1];
 
                            iP1 = iP - yOffsetP + xOffset;
                            ac_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];
                                        + rb[iR] * a_cse[iAm1] * pc[iP1]
                                        + rc[iR] * a_cse[iA] * pa[iP1];

                            iP1 = iP - xOffsetP;
                            ac_cw[iAc] = rc[iR] * a_cw[iA] * pc[iP1]
                                       + rb[iR] * a_cw[iAm1] * pb[iP1]
                                       + ra[iR] * a_cw[iAp1] * pa[iP1];
                                       + rb[iR] * a_cnw[iAm1] * pc[iP]
                                       + ra[iR] * a_csw[iAp1] * pc[iP]
                                       + rc[iR] * a_csw[iA] * pb[iP]
                                       + rc[iR] * a_cnw[iA] * pa[iP];
 
                            ac_cc[iAc] = rc[iR] * a_cc[iA] * pc[iP]
                                       + rb[iR] * a_cc[iAm1] * pb[iP]
                                       + ra[iR] * a_cc[iAp1] * pa[iP]
                                       + rb[iR] * a_cn[iAm1] * pc[iP]
                                       + ra[iR] * a_cs[iAp1] * pc[iP]
                                       + rc[iR] * a_cs[iA] * pb[iP]
                                       + rc[iR] * a_cn[iA] * pa[iP];

                         });

              break;

            }
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

zzz_SMG2BuildRAPNoSym(
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
 * Extract pointers for 5-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient
 * a_ce is pointer for east coefficient
 * a_cs is pointer for south coefficient
 * a_cn is pointer for north coefficient
 *--------------------------------------------------------------------------*/

            a_cc = ExtractPointerA(A,...);
            a_cw = ExtractPointerA(A,...);
            a_ce = ExtractPointerA(A,...);
            a_cs = ExtractPointerA(A,...);
            a_cn = ExtractPointerA(A,...);

/*--------------------------------------------------------------------------
 * Extract additional pointers for 9-point fine grid operator:
 *
 * a_csw is pointer for southwest coefficient in same plane
 * a_cse is pointer for southeast coefficient in same plane
 * a_cnw is pointer for northwest coefficient in same plane
 * a_cne is pointer for northeast coefficient in same plane
 *--------------------------------------------------------------------------*/

            if(fine_stencil_num_points > 5)
            {
              a_csw = ExtractPointerA(A,...);
              a_cse = ExtractPointerA(A,...);
              a_cnw = ExtractPointerA(A,...);
              a_cne = ExtractPointerA(A,...);
            }

/*--------------------------------------------------------------------------
 * Extract pointers for coarse grid operator - always 9-point:
 *
 * We build only the lower triangular part (plus diagonal).
 * 
 * ac_cc is pointer for center coefficient (etc.)
 *--------------------------------------------------------------------------*/

            ac_cc = ExtractPointerAc(Ac,...);
            ac_cw = ExtractPointerAc(Ac,...);
            ac_cs = ExtractPointerAc(Ac,...);
            ac_csw = ExtractPointerAc(Ac,...);
            ac_cse = ExtractPointerAc(Ac,...);

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

            yOffsetA = ; 
            xOffsetP = ;
            yOffsetP = ;

/*--------------------------------------------------------------------------
 * Switch statement to direct control to apropriate BoxLoop depending
 * on stencil size. Default is full 9-point.
 *--------------------------------------------------------------------------*/

            switch (fine_stencil_num_points)
            {

/*--------------------------------------------------------------------------
 * Loop for 5-point fine grid operator; produces upper triangular
 * part of 9-point coarse grid operator - excludes diagonal. 
 * stencil entries: (northeast, north, northwest, and east)
 *--------------------------------------------------------------------------*/

              case 5:

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP + yOffsetP + xOffset;
                            ac_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pc[iP1]
                                       + rc[iR] * a_cn[iA] * pb[iP1];
 
                            iP1 = iP + yOffsetP - xOffset;
                            ac_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

                            iP1 = iP + xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
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

              zzz_BoxLoop4(box,index,
                         data_box1, start1, stride1, iP,
                         data_box2, start2, stride2, iR,
                         data_box3, start3, stride3, iA,
                         data_box4, start4, stride4, iAc,
                         {
                            iAm1 = iA - yOffsetA;
                            iAp1 = iA + yOffsetA;

                            iP1 = iP + yOffsetP + xOffset;
                            ac_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];
                                        + ra[iR] * a_cne[iAp1] * pc[iP1]
                                        + rc[iR] * a_cne[iA] * pb[iP1];

                            iP1 = iP + yOffsetP;
                            ac_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                                       + ra[iR] * a_cn[iAp1] * pc[iP1]
                                       + rc[iR] * a_cn[iA] * pb[iP1];
 
                            iP1 = iP + yOffsetP - xOffset;
                            ac_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];
                                        + ra[iR] * a_cnw[iAp1] * pc[iP1]
                                        + rc[iR] * a_cnw[iA] * pb[iP1];

                            iP1 = iP e xOffsetP;
                            ac_ce[iAc] = rc[iR] * a_ce[iA] * pc[iP1]
                                       + rb[iR] * a_ce[iAm1] * pb[iP1]
                                       + ra[iR] * a_ce[iAp1] * pa[iP1];
                                       + rb[iR] * a_cne[iAm1] * pc[iP]
                                       + ra[iR] * a_cse[iAp1] * pc[iP]
                                       + rc[iR] * a_cse[iA] * pb[iP]
                                       + rc[iR] * a_cne[iA] * pa[iP];

                         });

              break;


            }
}


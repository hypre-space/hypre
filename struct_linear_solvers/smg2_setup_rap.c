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
 * hypre_SMG2CreateRAPOp 
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_SMG2CreateRAPOp( hypre_StructMatrix *R,
                       hypre_StructMatrix *A,
                       hypre_StructMatrix *PT,
                       hypre_StructGrid   *coarse_grid )
{
   hypre_StructMatrix    *RAP;

   hypre_Index           *RAP_stencil_shape;
   hypre_StructStencil   *RAP_stencil;
   int                    RAP_stencil_size;
   int                    RAP_stencil_dim;
   int                    RAP_num_ghost[] = {1, 1, 1, 1, 0, 0};

   int                    j, i;
   int                    stencil_rank;
 
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
            hypre_SetIndex(RAP_stencil_shape[stencil_rank],i,j,0);
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
       * in the standard lexicalgraphic ordering.
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
               hypre_SetIndex(RAP_stencil_shape[stencil_rank],i,j,0);
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
    * Set number of ghost points
    *-----------------------------------------------------------------------*/
   if (hypre_StructMatrixSymmetric(A))
   {
      RAP_num_ghost[1] = 0;
      RAP_num_ghost[3] = 0;
   }
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
 * I've written two routines - hypre_SMG2BuildRAPSym to build the
 * lower triangular part of RAP (including the diagonal) and
 * hypre_SMG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the
 * first routine would be called. With full storage both would need to
 * be called.
 *
 *--------------------------------------------------------------------------*/

int
hypre_SMG2BuildRAPSym( hypre_StructMatrix *A,
                       hypre_StructMatrix *PT,
                       hypre_StructMatrix *R,
                       hypre_StructMatrix *RAP,
                       hypre_Index         cindex,
                       hypre_Index         cstride )

{

   hypre_Index           index;

   hypre_StructStencil  *fine_stencil;
   int                   fine_stencil_size;

   hypre_StructGrid     *fgrid;
   int                  *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   int                  *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   int                   fi, ci;
   int                   loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *PT_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   double               *a_csw, *a_cse, *a_cnw;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  yOffsetA; 
   int                  xOffsetP; 
   int                  yOffsetP; 

   int                  ierr = 0;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

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

         hypre_SetIndex(index,0,1,0);
         pa = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);

         hypre_SetIndex(index,0,-1,0);
         pb = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);
 
         /*-----------------------------------------------------------------
          * Extract pointers for restriction operator:
          * ra is pointer for weight for f-point above c-point 
          * rb is pointer for weight for f-point below c-point 
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,1,0);
         ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

         hypre_SetIndex(index,0,-1,0);
         rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
 
         /*-----------------------------------------------------------------
          * Extract pointers for 5-point fine grid operator:
          * 
          * a_cc is pointer for center coefficient
          * a_cw is pointer for west coefficient
          * a_ce is pointer for east coefficient
          * a_cs is pointer for south coefficient
          * a_cn is pointer for north coefficient
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,-1,0,0);
         a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,1,0,0);
         a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,0,-1,0);
         a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,0,1,0);
         a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         /*-----------------------------------------------------------------
          * Extract additional pointers for 9-point fine grid operator:
          *
          * a_csw is pointer for southwest coefficient
          * a_cse is pointer for southeast coefficient
          * a_cnw is pointer for northwest coefficient
          * a_cne is pointer for northeast coefficient
          *-----------------------------------------------------------------*/

         if(fine_stencil_size > 5)
         {
            hypre_SetIndex(index,-1,-1,0);
            a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

            hypre_SetIndex(index,1,-1,0);
            a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

            hypre_SetIndex(index,-1,1,0);
            a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
         }

         /*-----------------------------------------------------------------
          * Extract pointers for coarse grid operator - always 9-point:
          *
          * We build only the lower triangular part (plus diagonal).
          * 
          * rap_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,0,0);
         rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,0,-1,0);
         rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,-1,0);
         rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,-1,0);
         rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         /*-----------------------------------------------------------------
          * Define offsets for fine grid stencil and interpolation
          *
          * In the BoxLoop below I assume iA and iP refer to data associated
          * with the point which we are building the stencil for. The below
          * Offsets are used in refering to data associated with other points. 
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,1,0);
         yOffsetA = hypre_BoxOffsetDistance(A_dbox,index); 
         yOffsetP = hypre_BoxOffsetDistance(PT_dbox,index); 
         hypre_SetIndex(index,1,0,0);
         xOffsetP = hypre_BoxOffsetDistance(PT_dbox,index); 

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
            hypre_BoxLoop4Begin(loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
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

            break;

            /*--------------------------------------------------------------
             * Loop for symmetric 9-point fine grid operator; produces a
             * symmetric 9-point coarse grid operator. We calculate only the
             * lower triangular stencil entries: (southwest, south, southeast,
             * west, and center).
             *--------------------------------------------------------------*/

            default:

            hypre_BoxGetSize(cgrid_box, loop_size);
            hypre_BoxLoop4Begin(loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
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

            break;

         } /* end switch statement */

      } /* end ForBoxI */

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SMG2BuildRAPNoSym( hypre_StructMatrix *A,
                         hypre_StructMatrix *PT,
                         hypre_StructMatrix *R,
                         hypre_StructMatrix *RAP,
                         hypre_Index         cindex,
                         hypre_Index         cstride )

{

   hypre_Index             index;

   hypre_StructStencil  *fine_stencil;
   int                   fine_stencil_size;
                        
   hypre_StructGrid     *fgrid;
   int                  *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   int                  *cgrid_ids;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   int                   fi, ci;
   int                   loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *PT_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_cc, *a_cw, *a_ce, *a_cn;
   double               *a_cse, *a_cnw, *a_cne;

   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   int                  iA, iAm1, iAp1;
   int                  iAc;
   int                  iP, iP1;
   int                  iR;

   int                  yOffsetA;
   int                  xOffsetP;
   int                  yOffsetP;

   int                  ierr = 0;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

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

         hypre_SetIndex(index,0,1,0);
         pa = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);

         hypre_SetIndex(index,0,-1,0);
         pb = hypre_StructMatrixExtractPointerByIndex(PT, fi, index);
 
         /*-----------------------------------------------------------------
          * Extract pointers for restriction operator:
          * ra is pointer for weight for f-point above c-point 
          * rb is pointer for weight for f-point below c-point 
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,1,0);
         ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

         hypre_SetIndex(index,0,-1,0);
         rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
 
         /*-----------------------------------------------------------------
          * Extract pointers for 5-point fine grid operator:
          * 
          * a_cc is pointer for center coefficient
          * a_cw is pointer for west coefficient
          * a_ce is pointer for east coefficient
          * a_cs is pointer for south coefficient
          * a_cn is pointer for north coefficient
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,0,0);
         a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,-1,0,0);
         a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,1,0,0);
         a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         hypre_SetIndex(index,0,1,0);
         a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         /*-----------------------------------------------------------------
          * Extract additional pointers for 9-point fine grid operator:
          *
          * a_csw is pointer for southwest coefficient
          * a_cse is pointer for southeast coefficient
          * a_cnw is pointer for northwest coefficient
          * a_cne is pointer for northeast coefficient
          *-----------------------------------------------------------------*/

         if(fine_stencil_size > 5)
         {
            hypre_SetIndex(index,1,-1,0);
            a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

            hypre_SetIndex(index,-1,1,0);
            a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

            hypre_SetIndex(index,1,1,0);
            a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

         }

         /*-----------------------------------------------------------------
          * Extract pointers for coarse grid operator - always 9-point:
          *
          * We build only the upper triangular part.
          *
          * rap_ce is pointer for east coefficient (etc.)
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,1,0,0);
         rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,0,1,0);
         rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,1,0);
         rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,1,0);
         rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         /*-----------------------------------------------------------------
          * Define offsets for fine grid stencil and interpolation
          *
          * In the BoxLoop below I assume iA and iP refer to data associated
          * with the point which we are building the stencil for. The below
          * Offsets are used in refering to data associated with other points. 
          *-----------------------------------------------------------------*/

         hypre_SetIndex(index,0,1,0);
         yOffsetA = hypre_BoxOffsetDistance(A_dbox,index); 
         yOffsetP = hypre_BoxOffsetDistance(PT_dbox,index); 
         hypre_SetIndex(index,1,0,0);
         xOffsetP = hypre_BoxOffsetDistance(PT_dbox,index); 

         /*-----------------------------------------------------------------
          * Switch statement to direct control to apropriate BoxLoop depending
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
            hypre_BoxLoop4Begin(loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
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

            break;

            /*--------------------------------------------------------------
             * Loop for 9-point fine grid operator; produces upper triangular
             * part of 9-point coarse grid operator - excludes diagonal.
             * stencil entries: (northeast, north, northwest, and east)
             *--------------------------------------------------------------*/

            default:

            hypre_BoxGetSize(cgrid_box, loop_size);
            hypre_BoxLoop4Begin(loop_size,
                                PT_dbox,  cstart, stridec, iP,
                                R_dbox,   cstart, stridec, iR,
                                A_dbox,   fstart, stridef, iA,
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

            break;

         } /* end switch statement */

      } /* end ForBoxI */

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_SMG2RAPPeriodicSym 
 *    Collapses stencil in periodic direction on coarsest grid.
 *--------------------------------------------------------------------------*/
 

int
hypre_SMG2RAPPeriodicSym( hypre_StructMatrix *RAP,
                          hypre_Index         cindex,
                          hypre_Index         cstride )

{
   hypre_Index             index;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             loop_size;

   int                  ci;
   int                  loopi, loopj, loopk;

   hypre_Box              *RAP_dbox;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;

   int                  iAc;
   int                  iAcm1;

   int                  xOffset;

   double               zero = 0.0;

   int                  ierr = 0;

   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   if (hypre_IndexY(hypre_StructGridPeriodic(cgrid)) == 1)
   {
      hypre_StructMatrixAssemble(RAP);
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
   
         cstart = hypre_BoxIMin(cgrid_box);

         RAP_dbox =
         hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

         hypre_SetIndex(index,1,0,0);
         xOffset = hypre_BoxOffsetDistance(RAP_dbox,index); 

         /*-----------------------------------------------------------------
          * Extract pointers for coarse grid operator - always 9-point:
          *-----------------------------------------------------------------*/
         hypre_SetIndex(index,0,0,0);
         rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,0,0);
         rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,0,-1,0);
         rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,-1,0);
         rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,-1,0);
         rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_BoxGetSize(cgrid_box, loop_size);


         hypre_BoxLoop1Begin(loop_size,
                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc,iAcm1
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
            {
               iAcm1 = iAc - xOffset;
               
               rap_cw[iAc] += (rap_cse[iAcm1] + rap_csw[iAc]);
               rap_cc[iAc] += (2.0 * rap_cs[iAc]);
            }
         hypre_BoxLoop1End(iAc);

         hypre_BoxLoop1Begin(loop_size,
                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
            {
               rap_csw[iAc] = zero;
               rap_cs[iAc] = zero;
               rap_cse[iAc] = zero;
            }
         hypre_BoxLoop1End(iAc);

      } /* end ForBoxI */

   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMG2RAPPeriodicNoSym 
 *    Collapses stencil in periodic direction on coarsest grid.
 *--------------------------------------------------------------------------*/
 
int
hypre_SMG2RAPPeriodicNoSym( hypre_StructMatrix *RAP,
                            hypre_Index         cindex,
                            hypre_Index         cstride )

{

   hypre_Index             index;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_IndexRef          cstart;
   hypre_Index             stridec;
   hypre_Index             loop_size;

   int                  ci;
   int                  loopi, loopj, loopk;

   hypre_Box              *RAP_dbox;

   double               *rap_cc, *rap_cw, *rap_cs;
   double               *rap_csw, *rap_cse;
   double               *rap_ce, *rap_cn;
   double               *rap_cnw, *rap_cne;

   int                  iAc;

   double               zero = 0.0;

   int                  ierr = 0;

   hypre_SetIndex(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   if (hypre_IndexY(hypre_StructGridPeriodic(cgrid)) == 1)
   {
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
   
         cstart = hypre_BoxIMin(cgrid_box);

         RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

         /*-----------------------------------------------------------------
          * Extract pointers for coarse grid operator - always 9-point:
          *-----------------------------------------------------------------*/
         hypre_SetIndex(index,0,0,0);
         rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,0,0);
         rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,0,-1,0);
         rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,-1,0);
         rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,-1,0);
         rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,0,0);
         rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,0,1,0);
         rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,1,1,0);
         rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

         hypre_SetIndex(index,-1,1,0);
         rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);


         hypre_BoxGetSize(cgrid_box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
            {
               rap_cw[iAc] += (rap_cnw[iAc] + rap_csw[iAc]);
               rap_cnw[iAc] = zero;
               rap_csw[iAc] = zero;
               
               rap_cc[iAc] += (rap_cn[iAc] + rap_cs[iAc]);
               rap_cn[iAc] = zero;
               rap_cs[iAc] = zero;
               
               rap_ce[iAc] += (rap_cne[iAc] + rap_cse[iAc]);
               rap_cne[iAc] = zero;
               rap_cse[iAc] = zero;
            }
         hypre_BoxLoop1End(iAc);

      } /* end ForBoxI */

   }
   return ierr;
}


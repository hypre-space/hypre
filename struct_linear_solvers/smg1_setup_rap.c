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
 * zzz_SMG1NewRAPOp 
 *    Sets up new coarse grid operator stucture.
 *    Differs from 2 and 3D verions in that R and PT are not in the
 *    argument list and coarse_grid is.
 *--------------------------------------------------------------------------*/
 
zzz_StructMatrix *
zzz_SMG1NewRAPOp( zzz_StructMatrix *A,
                  zzz_StructGrid   *coarse_grid )
{
   zzz_StructMatrix    *RAP;

   zzz_Index          **RAP_stencil_shape;
   zzz_StructStencil   *RAP_stencil;
   int                  RAP_stencil_size;
   int                  RAP_stencil_dim;
   int                  RAP_num_ghost[2];

   int                  i;
   int                  stencil_rank;
 
   RAP_stencil_dim = 1;

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
 *    3 point fine grid stencil produces 3 point RAP
 *--------------------------------------------------------------------------*/
      RAP_stencil_size = 3;
      RAP_stencil_shape = zzz_CTAlloc(zzz_Index *, RAP_stencil_size);
      for (i = -1; i < 2; i++)
      {

/*--------------------------------------------------------------------------
 *           Storage for 3 elements (c,w,e)
 *--------------------------------------------------------------------------*/
         RAP_stencil_shape[stencil_rank] = zzz_NewIndex();
         zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,0,0);
         stencil_rank++;
      }
   }

/*--------------------------------------------------------------------------
 * symmetric case
 *--------------------------------------------------------------------------*/

   else
   {

/*--------------------------------------------------------------------------
 *    3 point fine grid stencil produces 3 point RAP
 *    Only store the lower triangular part + diagonal = 2 entries,
 *    lower triangular means the lower triangular part on the matrix
 *    in the standard lexicalgraphic ordering.
 *--------------------------------------------------------------------------*/
      RAP_stencil_size = 2;
      RAP_stencil_shape = zzz_CTAlloc(zzz_Index *, RAP_stencil_size);
      for (i = -1; i < 1; i++)
      {

/*--------------------------------------------------------------------------
 *           Store 2 elements in (c,w)
 *--------------------------------------------------------------------------*/
         RAP_stencil_shape[stencil_rank] = zzz_NewIndex();
         zzz_SetIndex(RAP_stencil_shape[stencil_rank],i,0,0);
         stencil_rank++;
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
   for (i = 0; i < 2; i++ )
       RAP_num_ghost[i] = 1;
   zzz_SetStructMatrixNumGhost(RAP, RAP_num_ghost);

   zzz_InitializeStructMatrix(RAP);
 
   return RAP;
}

/*--------------------------------------------------------------------------
 * Routine to build 1-D RAP.
 *
 * Differs from 2 and 3-D versions in 2 major ways
 *
 *  1) Single routine zzz_SMG1BuildRAP handes both symmetric and
 *     non-symmetric case.
 *
 *  2) Interpolation and restriction weigths are calculated as
 *     needed - they are not in the argument list.
 *
 * I assume that the c-to-c interpolation is the identity.
 *
 *--------------------------------------------------------------------------*/

int
zzz_SMG1BuildRAP( zzz_StructMatrix *A,
                  zzz_StructMatrix *RAP )

{

   zzz_Index            *index_temp;

   zzz_Index            *cindex;
   zzz_Index            *cstride;

   zzz_StructGrid       *cgrid;
   zzz_BoxArray         *cgrid_boxes;
   zzz_Box              *cgrid_box;
   zzz_Index            *cstart;
   zzz_Index            *stridec;
   zzz_Index            *fstart;
   zzz_Index            *stridef;
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;

   int                  i;

   zzz_Box              *A_data_box;
   zzz_Box              *RAP_data_box;

   double               *a_cc, *a_cw, *a_ce;
   double               *rap_cc, *rap_cw, *rap_ce;

   int                  iA, iAm1, iAp1;
   int                  iAc;

   int                  xOffsetA; 

   int                  ierr;

   index_temp = zzz_NewIndex();
   loop_index = zzz_NewIndex();
   loop_size = zzz_NewIndex();

   stridef = cstride;
   fstart = zzz_NewIndex();
   stridec = zzz_NewIndex();
   zzz_SetIndex(stridec, 1, 0, 0);

   cgrid = zzz_StructMatrixGrid(RAP);
   cgrid_boxes = zzz_StructGridBoxes(cgrid);

   zzz_ForBoxI(i, cgrid_boxes)
   {
      cgrid_box = zzz_BoxArrayBox(cgrid_boxes, i);

      cstart = zzz_BoxIMin(cgrid_box);
      zzz_SMGMapCoarseToFine(cstart, fstart, cindex, cstride) 

      A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
      RAP_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(RAP), i);

/*--------------------------------------------------------------------------
 * Extract pointers for 3-point fine grid operator:
 * 
 * a_cc is pointer for center coefficient
 * a_cw is pointer for west coefficient
 * a_ce is pointer for east coefficient
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      a_cc = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      a_cw = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

      zzz_SetIndex(index_temp,1,0,0);
      a_ce = zzz_StructMatrixExtractPointerByIndex(A, i, index_temp);

/*--------------------------------------------------------------------------
 * Extract pointers for coarse grid operator - always 3-point:
 *
 * If A is symmetric so will RAP, we build only the lower triangular part
 * (plus diagonal).
 * 
 * rap_cc is pointer for center coefficient (etc.)
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,0,0,0);
      rap_cc = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      zzz_SetIndex(index_temp,-1,0,0);
      rap_cw = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);

      if(!zzz_StructMatrixSymmetric(A))
      {
         zzz_SetIndex(index_temp,0,1,0);
         rap_ce = zzz_StructMatrixExtractPointerByIndex(RAP, i, index_temp);
      }

/*--------------------------------------------------------------------------
 * Define offsets for fine grid stencil and interpolation
 *
 * In the BoxLoop below I assume iA and iP refer to data associated
 * with the point which we are building the stencil for. The below
 * Offsets are used in refering to data associated with other points. 
 *--------------------------------------------------------------------------*/

      zzz_SetIndex(index_temp,1,0,0);
      xOffsetA = zzz_BoxOffsetDistance(A_data_box,index_temp); 

/*--------------------------------------------------------------------------
 * non-symmetric case
 *--------------------------------------------------------------------------*/

      if(!zzz_StructMatrixSymmetric(A))
      {
         zzz_GetBoxSize(cgrid_box, loop_size);
         zzz_BoxLoop2(loop_index, loop_size,
                      A_data_box, fstart, stridef, iA,
                      RAP_data_box, cstart, stridec, iAc,
                      {
                         iAm1 = iA - xOffsetA;
                         iAp1 = iA + xOffsetA;

                         rap_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                         rap_cc[iAc] = a_cc[iA]
                                     - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                                     - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   

                         rap_ce[iAc] = - a_ce[iA] *a_ce[iAp1] / a_cc[iAp1];

                      });
      }

/*--------------------------------------------------------------------------
 * symmetric case
 *--------------------------------------------------------------------------*/

      else
      {
         zzz_GetBoxSize(cgrid_box, loop_size);
         zzz_BoxLoop2(loop_index, loop_size,
                      A_data_box, fstart, stridef, iA,
                      RAP_data_box, cstart, stridec, iAc,
                      {
                         iAm1 = iA - xOffsetA;
                         iAp1 = iA + xOffsetA;

                         rap_cw[iAc] = - a_cw[iA] *a_cw[iAm1] / a_cc[iAm1];

                         rap_cc[iAc] = a_cc[iA]
                                     - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1]   
                                     - a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];   
                      });
      }

   } /* end ForBoxI */

   zzz_FreeIndex(index_temp);
   zzz_FreeIndex(stridec);
   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(fstart);

   return ierr;
}

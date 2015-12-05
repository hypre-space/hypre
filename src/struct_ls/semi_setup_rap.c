/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

#define hypre_MapRAPMarker(indexRAP, rank) \
{\
   HYPRE_Int imacro,jmacro,kmacro;\
   imacro = hypre_IndexX(indexRAP);\
   jmacro = hypre_IndexY(indexRAP);\
   kmacro = hypre_IndexZ(indexRAP);\
   if (imacro==-1) imacro=2;\
   if (jmacro==-1) jmacro=2;\
   if (kmacro==-1) kmacro=2;\
   rank = imacro + 3*jmacro + 9*kmacro;\
}

#define hypre_InverseMapRAPMarker(rank, indexRAP) \
{\
   HYPRE_Int imacro,ijmacro,jmacro,kmacro;\
   ijmacro = (rank%9);\
   imacro  = (ijmacro%3);\
   jmacro  = (ijmacro-imacro)/3;\
   kmacro  = (rank-3*jmacro-imacro)/9;\
   if (imacro==2) imacro=-1;\
   if (jmacro==2) jmacro=-1;\
   if (kmacro==2) kmacro=-1;\
   hypre_SetIndex(indexRAP,imacro,jmacro,kmacro);\
}

/*--------------------------------------------------------------------------
 * hypre_SemiCreateRAPOp 
 *    Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_SemiCreateRAPOp( hypre_StructMatrix *R,
                       hypre_StructMatrix *A,
                       hypre_StructMatrix *P,
                       hypre_StructGrid   *coarse_grid,
                       HYPRE_Int           cdir,
                       HYPRE_Int           P_stored_as_transpose )
{
   hypre_StructMatrix    *RAP;

   hypre_Index           *RAP_stencil_shape;
   hypre_StructStencil   *RAP_stencil;
   HYPRE_Int              RAP_stencil_size;
   HYPRE_Int              dim;
   HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   HYPRE_Int             *not_cdirs;
   hypre_StructStencil   *A_stencil;
   HYPRE_Int              A_stencil_size;
   hypre_Index           *A_stencil_shape;

   hypre_Index            indexR;
   hypre_Index            indexRA;
   hypre_Index            indexRAP;
   HYPRE_Int              Rloop, Aloop;

   HYPRE_Int              j, i;
   HYPRE_Int              d;
   HYPRE_Int              stencil_rank;

   HYPRE_Int             *RAP_marker;
   HYPRE_Int              RAP_marker_size;
   HYPRE_Int              RAP_marker_rank;

   A_stencil = hypre_StructMatrixStencil(A);
   dim = hypre_StructStencilDim(A_stencil);
   A_stencil_size = hypre_StructStencilSize(A_stencil);
   A_stencil_shape = hypre_StructStencilShape(A_stencil);
 
   /*-----------------------------------------------------------------------
    * Allocate RAP_marker array used to deternine which offsets are
    * present in RAP. Initialized to zero indicating no offsets present.
    *-----------------------------------------------------------------------*/

   RAP_marker_size = 1;
   for (i = 0; i < dim; i++)
   {
      RAP_marker_size *= 3;
   }
   RAP_marker = hypre_CTAlloc(HYPRE_Int, RAP_marker_size);
   
   /*-----------------------------------------------------------------------
    * Define RAP_stencil
    *-----------------------------------------------------------------------*/

   hypre_ClearIndex(indexR);
   hypre_ClearIndex(indexRA);
   hypre_ClearIndex(indexRAP);

   stencil_rank = 0;

   /*-----------------------------------------------------------------------
    * Calculate RAP stencil by symbolic computation of triple matrix
    * product RAP. We keep track of index to update RAP_marker.
    *-----------------------------------------------------------------------*/
   for (Rloop = -1; Rloop < 2; Rloop++)
   {
      hypre_IndexD(indexR,cdir) = Rloop;
      for (Aloop = 0; Aloop < A_stencil_size; Aloop++)
      {
         for (d = 0; d < dim; d++)
         {
            hypre_IndexD(indexRA, d) = hypre_IndexD(indexR, d) +
                                       hypre_IndexD(A_stencil_shape[Aloop], d);  
         }
         
         /*-----------------------------------------------------------------
          * If RA part of the path lands on C point, then P part of path
          * stays at the C point. Divide by 2 to yield to coarse index.  
          *-----------------------------------------------------------------*/
         if ((hypre_IndexD(indexRA, cdir) % 2) == 0)
         {
            hypre_CopyIndex(indexRA, indexRAP);
            hypre_IndexD(indexRAP,cdir) /= 2;
            hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;
         }
         /*-----------------------------------------------------------------
          * If RA part of the path lands on F point, then P part of path
          * move +1 and -1 in cdir. Divide by 2 to yield to coarse index.  
          *-----------------------------------------------------------------*/
         else
         {
            hypre_CopyIndex(indexRA, indexRAP);
            hypre_IndexD(indexRAP,cdir) += 1;
            hypre_IndexD(indexRAP,cdir) /= 2;
            hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;

            hypre_CopyIndex(indexRA, indexRAP);
            hypre_IndexD(indexRAP,cdir) -= 1;
            hypre_IndexD(indexRAP,cdir) /= 2;
            hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    * For symmetric A, we zero out some entries of RAP_marker to yield
    * the stencil with the proper stored entries.
    * The set S of stored off diagonal entries are such that paths in
    * RAP resulting in a contribution to a entry of S arise only from
    * diagonal entries of A or entries contined in S.
    * 
    * In 1d
    * =====
    * cdir = 0       
    * (i) in S if   
    *    i<0.
    * 
    * In 2d
    * =====
    * cdir = 1                 cdir = 0    
    * (i,j) in S if          (i,j) in S if      
    *      i<0,                     j<0,       
    * or   i=0 & j<0.          or   j=0 & i<0.
    * 
    * In 3d
    * =====
    * cdir = 2                 cdir = 1                cdir = 0
    * (i,j,k) in S if          (i,j,k) in S if         (i,j,k) in S if
    *      i<0,                     k<0,                    j<0,
    * or   i=0 & j<0,          or   k=0 & i<0,              j=0 & k<0,
    * or   i=j=0 & k<0.        or   k=i=0 & j<0.            j=k=0 & i<0.
    *-----------------------------------------------------------------------*/
   if (hypre_StructMatrixSymmetric(A))
   {
      if (dim > 1)
      {
         not_cdirs = hypre_CTAlloc(HYPRE_Int, dim-1);
      }

      for (d = 1; d < dim; d++)
      {
         not_cdirs[d-1] = (dim+cdir-d) % dim;
      }

      hypre_ClearIndex(indexRAP);
      hypre_IndexD(indexRAP, cdir) = 1;
      hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
      RAP_marker[RAP_marker_rank] = 0;

      if (dim > 1)
      {
         hypre_ClearIndex(indexRAP);
         hypre_IndexD(indexRAP,not_cdirs[0]) = 1;
         for (i = -1; i < 2; i++)
         {
            hypre_IndexD(indexRAP,cdir) = i;
            hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
            RAP_marker[RAP_marker_rank] = 0;
         }
      }
   
      if (dim > 2)
      {
         hypre_ClearIndex(indexRAP);
         hypre_IndexD(indexRAP,not_cdirs[1]) = 1;
         for (i = -1; i < 2; i++)
         {
            hypre_IndexD(indexRAP,not_cdirs[0]) = i;
            for (j = -1; j < 2; j++)
            {
               hypre_IndexD(indexRAP,cdir) = j;
               hypre_MapRAPMarker(indexRAP,RAP_marker_rank);
               RAP_marker[RAP_marker_rank] = 0;

            }
         }
      }

      if (dim > 1)
      {
         hypre_TFree(not_cdirs);
      }
   }

   RAP_stencil_size= 0;

   for (i = 0; i < RAP_marker_size; i++)
   {
      if ( RAP_marker[i] != 0 )
      {
         RAP_stencil_size++;
      }
   }

   RAP_stencil_shape = hypre_CTAlloc(hypre_Index, RAP_stencil_size);

   stencil_rank= 0;
   for (i = 0; i < RAP_marker_size; i++)
   {
      if ( RAP_marker[i] != 0 )
      {
         hypre_InverseMapRAPMarker(i,RAP_stencil_shape[stencil_rank]);
         stencil_rank++;
      }
   }

   RAP_stencil = hypre_StructStencilCreate(dim, RAP_stencil_size,
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

   hypre_TFree(RAP_marker);

   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_SemiBuildRAP
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SemiBuildRAP( hypre_StructMatrix *A,
                    hypre_StructMatrix *P,
                    hypre_StructMatrix *R,
                    HYPRE_Int           cdir,
                    hypre_Index         cindex,
                    hypre_Index         cstride,
                    HYPRE_Int           P_stored_as_transpose,
                    hypre_StructMatrix *RAP     )
{

   hypre_Index           index;

   hypre_StructStencil  *coarse_stencil;
   HYPRE_Int             coarse_stencil_size;
   hypre_Index          *coarse_stencil_shape;
   HYPRE_Int            *coarse_symm_elements;

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
   HYPRE_Int             loopi, loopj, loopk;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   double               *pa, *pb;
   double               *ra, *rb;

   double               *a_ptr;

   double               *rap_ptrS, *rap_ptrU, *rap_ptrD;

   HYPRE_Int             symm_path_multiplier;

   HYPRE_Int             iA, iAp;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iPp;
   HYPRE_Int             iR;
                        
   HYPRE_Int             COffsetA; 
   HYPRE_Int             COffsetP; 
   HYPRE_Int             AOffsetP; 

   HYPRE_Int             RAPloop;
   HYPRE_Int             diag;
   HYPRE_Int             dim;
   HYPRE_Int             d;
                     
   double                zero = 0.0;
   HYPRE_Int             ierr = 0;

   coarse_stencil = hypre_StructMatrixStencil(RAP);
   coarse_stencil_size = hypre_StructStencilSize(coarse_stencil);
   coarse_symm_elements = hypre_StructMatrixSymmElements(RAP);
   coarse_stencil_shape = hypre_StructStencilShape(coarse_stencil);
   dim = hypre_StructStencilDim(coarse_stencil);

   stridef = cstride;
   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   /*-----------------------------------------------------------------
    *  Loop over boxes to compute entries of RAP
    *-----------------------------------------------------------------*/

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
         hypre_BoxGetSize(cgrid_box, loop_size);

         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
         P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
         R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
         RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

         /*-----------------------------------------------------------------
          * Extract pointers for interpolation operator:
          * pa is pointer for weight for f-point above c-point 
          * pb is pointer for weight for f-point below c-point 
          *
          *   pa  "down"                      pb "up"
          *
          *                                     C
          *
          *                                     |  
          *                                     v
          *
          *       F                             F                             
          *
          *       ^
          *       |
          *
          *       C
          *
          *-----------------------------------------------------------------*/

         hypre_ClearIndex(index);
         if (P_stored_as_transpose)
         {
            hypre_IndexD(index, cdir) = 1;
            pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
            
            hypre_IndexD(index, cdir) = -1;
            pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
         }
         else
         {
            hypre_IndexD(index, cdir) = -1;
            pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

            hypre_IndexD(index, cdir) = 1;
            pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index) -
              hypre_BoxOffsetDistance(P_dbox, index);
         }
 
         /*-----------------------------------------------------------------
          * Extract pointers for restriction operator:
          * ra is pointer for weight for f-point above c-point 
          * rb is pointer for weight for f-point below c-point 
          *
          *   rb  "down"                      ra "up"
          *
          *                                     F
          *
          *                                     |  
          *                                     v
          *
          *       C                             C                             
          *
          *       ^
          *       |
          *
          *       F
          *
          *-----------------------------------------------------------------*/

         hypre_ClearIndex(index);
         if (P_stored_as_transpose)
         {
            hypre_IndexD(index, cdir) = 1;
            ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
            
            hypre_IndexD(index, cdir) = -1;
            rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
         }
         else
         {
            hypre_IndexD(index, cdir) = -1;
            ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

            hypre_IndexD(index, cdir) = 1;
            rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index) -
              hypre_BoxOffsetDistance(P_dbox, index);
         }
 
         /*-----------------------------------------------------------------
          * Define offsets for fine grid stencil and interpolation
          *
          * In the BoxLoops below I assume iA and iP refer to data associated
          * with the point which we are building the stencil for. The below
          * Offsets (and those defined later in the switch statement) are
          * used in refering to data associated with other points. 
          *-----------------------------------------------------------------*/

         hypre_ClearIndex(index);
         hypre_IndexD(index, cdir) = 1;
         COffsetA = hypre_BoxOffsetDistance(A_dbox,index);
         COffsetP = hypre_BoxOffsetDistance(P_dbox,index);

         /*-----------------------------------------------------------------
          * Entries in RAP are calculated by accumulation, must first
          * zero out entries.
          *-----------------------------------------------------------------*/

         for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
         {
            if (coarse_symm_elements[RAPloop] == -1)
            {
               rap_ptrS = hypre_StructMatrixBoxData(RAP, ci, RAPloop);
               hypre_BoxLoop1Begin(loop_size,
                                   RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
                  {
                     rap_ptrS[iAc] = zero;
                  }
               hypre_BoxLoop1End(iAc);
            }
         }

         /*-----------------------------------------------------------------
          * Computational loop. Written as a loop over stored entries of
          * RAP. We then get the pointer (a_ptr) for the same index in A.
          * If it exists, we then calculate all RAP paths involving this
          * entry of A.
          *-----------------------------------------------------------------*/
         for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
         {
            if (coarse_symm_elements[RAPloop] == -1)
            {
               /*-------------------------------------------------------------
                * Get pointer for A that corresponds to the current RAP index.
                * If pointer is non-null, i.e. there is a corresponding entry
                * in A, compute paths.
                *-------------------------------------------------------------*/
               hypre_CopyIndex(coarse_stencil_shape[RAPloop], index);
               a_ptr = hypre_StructMatrixExtractPointerByIndex(A, fi, index);
               if (a_ptr != NULL)
               {
                  switch (hypre_IndexD(index, cdir))
                  {
                     /*-----------------------------------------------------
                      * If A stencil index is 0 in coarsened direction, need
                      * to calculate (r,p) pairs (stay,stay) (up,up) (up,down)
                      * (down,up) and (down,down). Paths 1,3 & 4 {(s,s),(u,d),
                      * (d,u)} yield contributions to RAP with the same stencil
                      * index as A. Path 2 (u,u) contributes to RAP with 
                      * index +1 in coarsened direction. Path 5 (d,d)
                      * contributes to RAP with index -1 in coarsened
                      * direction.
                      *-----------------------------------------------------*/
 
                      case 0:

                      hypre_IndexD(index,cdir) = 1;
                      rap_ptrU = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      hypre_IndexD(index,cdir) = -1;
                      rap_ptrD = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      hypre_IndexD(index,cdir) = 0;
                      AOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
                      rap_ptrS = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      diag = 0;
                      for (d = 0; d < dim; d++)
                      {
                         diag += hypre_IndexD(index,d) * hypre_IndexD(index,d);
                      }
                
                      if (diag == 0 && hypre_StructMatrixSymmetric(RAP))
                      {  
                         /*--------------------------------------------------
                          * If A stencil index is (0,0,0) and RAP is symmetric,
                          * must not calculate (up,up) path. It's symmetric
                          * to the (down,down) path and calculating both paths
                          * incorrectly doubles the contribution. Additionally
                          * the (up,up) path contributes to a non-stored entry
                          * in RAP.
                          *--------------------------------------------------*/
                         hypre_BoxLoop4Begin(loop_size,
                                             P_dbox, cstart, stridec, iP,
                                             R_dbox, cstart, stridec, iR,
                                             A_dbox, fstart, stridef, iA,
                                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAp,iPp
#include "hypre_box_smp_forloop.h"
                         hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
                            {
                               /* path 1 : (stay,stay) */
                               rap_ptrS[iAc] +=          a_ptr[iA]           ;

                               /* path 2 : (up,up) */

                               /* path 3 : (up,down) */
                               iAp = iA + COffsetA;
                               iPp = iP + AOffsetP; 
                               rap_ptrS[iAc] += ra[iR] * a_ptr[iAp] * pa[iPp];

                               /* path 4 : (down,up) */
                               iAp = iA - COffsetA;
                               rap_ptrS[iAc] += rb[iR] * a_ptr[iAp] * pb[iPp];

                               /* path 5 : (down,down) */
                               iPp = iP - COffsetP + AOffsetP; 
                               rap_ptrD[iAc] += rb[iR] * a_ptr[iAp] * pa[iPp];
                            }
                         hypre_BoxLoop4End(iP, iR, iA, iAc);
                      }
                      else
                      {
                         /*--------------------------------------------------
                          * If A stencil index is not (0,0,0) or RAP is
                          * nonsymmetric, all 5 paths are calculated.
                          *--------------------------------------------------*/
                         hypre_BoxLoop4Begin(loop_size,
                                             P_dbox, cstart, stridec, iP,
                                             R_dbox, cstart, stridec, iR,
                                             A_dbox, fstart, stridef, iA,
                                             RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAp,iPp
#include "hypre_box_smp_forloop.h"
                         hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
                            {
                               /* path 1 : (stay,stay) */
                               rap_ptrS[iAc] +=          a_ptr[iA]           ;

                               /* path 2 : (up,up) */
                               iAp = iA + COffsetA;
                               iPp = iP + COffsetP + AOffsetP; 
                               rap_ptrU[iAc] += ra[iR] * a_ptr[iAp] * pb[iPp];

                               /* path 3 : (up,down) */
                               iPp = iP + AOffsetP; 
                               rap_ptrS[iAc] += ra[iR] * a_ptr[iAp] * pa[iPp];

                               /* path 4 : (down,up) */
                               iAp = iA - COffsetA;
                               rap_ptrS[iAc] += rb[iR] * a_ptr[iAp] * pb[iPp];

                               /* path 5 : (down,down) */
                               iPp = iP - COffsetP + AOffsetP; 
                               rap_ptrD[iAc] += rb[iR] * a_ptr[iAp] * pa[iPp];
                            }
                         hypre_BoxLoop4End(iP, iR, iA, iAc);
                      }

                      break;

                     /*-----------------------------------------------------
                      * If A stencil index is -1 in coarsened direction, need
                      * to calculate (r,p) pairs (stay,up) (stay,down) (up,stay)
                      * and (down,stay). Paths 2 & 4 {(s,d),(d,s)} contribute
                      * to RAP with same stencil index as A. Paths 1 & 3 
                      * {(s,u),(u,s)} contribute to RAP with index 0 in 
                      * coarsened direction.
                      *-----------------------------------------------------*/
 
                      case -1:

                      rap_ptrD = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      hypre_IndexD(index,cdir) = 0;
                      AOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
                      rap_ptrS = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);

                      /*--------------------------------------------------
                       * If A stencil index is zero except in coarsened
                       * dirction and RAP is symmetric, must calculate
                       * symmetric paths for (stay,up) and (up,stay).
                       * These contribute to the diagonal entry of RAP.
                       * These additional paths have the same numerical
                       * contribution as the calculated path. We multiply
                       * by two to account for them.
                       *--------------------------------------------------*/
                      symm_path_multiplier = 1;
                      diag = 0;
                      for (d = 0; d < dim; d++)
                      {
                         diag += hypre_IndexD(index,d) * hypre_IndexD(index,d);
                      }
                      if (diag == 0 && hypre_StructMatrixSymmetric(RAP))
                      {
                         symm_path_multiplier = 2;
                      }

                      hypre_BoxLoop4Begin(loop_size,
                                          P_dbox, cstart, stridec, iP,
                                          R_dbox, cstart, stridec, iR,
                                          A_dbox, fstart, stridef, iA,
                                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAp,iPp
#include "hypre_box_smp_forloop.h"
                      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
                         {
                            /* Path 1 : (stay,up) & symmetric path  */
                            iPp = iP + AOffsetP; 
                            rap_ptrS[iAc] += symm_path_multiplier *
                                                     (a_ptr[iA]  * pb[iPp]);

                            /* Path 2 : (stay,down) */
                            iPp = iP - COffsetP + AOffsetP; 
                            rap_ptrD[iAc] +=          a_ptr[iA]  * pa[iPp];

                            /* Path 3 : (up,stay) */
                            iAp = iA + COffsetA;
                            rap_ptrS[iAc] += symm_path_multiplier *
                                            (ra[iR] * a_ptr[iAp]          );

                            /* Path 4 : (down,stay) */
                            iAp = iA - COffsetA;
                            rap_ptrD[iAc] += rb[iR] * a_ptr[iAp]          ;
                         }
                      hypre_BoxLoop4End(iP, iR, iA, iAc);

                      break;

                     /*-----------------------------------------------------
                      * If A stencil index is +1 in coarsened direction, need
                      * to calculate (r,p) pairs (stay,up) (stay,down) (up,stay)
                      * and (down,stay). Paths 1 & 3 {(s,u),(u,s)} contribute
                      * to RAP with same stencil index as A. Paths 2 & 4 
                      * {(s,d),(d,s)} contribute to RAP with index 0 in
                      * coarsened direction.
                      *-----------------------------------------------------*/

                      case 1:

                      rap_ptrU = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      hypre_IndexD(index,cdir) = 0;
                      AOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
                      rap_ptrS = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      /*--------------------------------------------------
                       * If A stencil index is zero except in coarsened
                       * dirction and RAP is symmetric, must calculate
                       * symmetric paths for (stay,down) and (down,stay).
                       * These contribute to the diagonal entry of RAP.
                       * These additional paths have the same numerical
                       * contribution as the calculated path. We multiply
                       * by two to account for them.
                       *--------------------------------------------------*/
                      symm_path_multiplier = 1;
                      diag = 0;
                      for (d = 0; d < dim; d++)
                      {
                         diag += hypre_IndexD(index,d) * hypre_IndexD(index,d);
                      }
                      if (diag == 0 && hypre_StructMatrixSymmetric(RAP))
                      {
                         symm_path_multiplier = 2;
                      }

                      hypre_BoxLoop4Begin(loop_size,
                                          P_dbox, cstart, stridec, iP,
                                          R_dbox, cstart, stridec, iR,
                                          A_dbox, fstart, stridef, iA,
                                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iP,iR,iA,iAc,iAp,iPp
#include "hypre_box_smp_forloop.h"
                      hypre_BoxLoop4For(loopi, loopj, loopk, iP, iR, iA, iAc)
                         {
                            /* Path 1 : (stay,up) */
                            iPp = iP + COffsetP + AOffsetP; 
                            rap_ptrU[iAc] +=          a_ptr[iA]  * pb[iPp];

                            /* Path 2 : (stay,down) */
                            iPp = iP + AOffsetP; 
                            rap_ptrS[iAc] += symm_path_multiplier *
                                                     (a_ptr[iA]  * pa[iPp]);

                            /* Path 3 : (up,stay) */
                            iAp = iA + COffsetA;
                            rap_ptrU[iAc] += ra[iR] * a_ptr[iAp]          ;

                            /* Path 4 : (down,stay) */
                            iAp = iA - COffsetA;
                            rap_ptrS[iAc] += symm_path_multiplier *
                                            (rb[iR] * a_ptr[iAp]          );
                         }
                      hypre_BoxLoop4End(iP, iR, iA, iAc);

                      break;
                  } /* end of switch */

               } /* end of if a_ptr != NULL */

            } /* end if coarse_symm_element == -1 */

         } /* end of RAPloop */

      } /* end ForBoxI */

   /*-----------------------------------------------------------------
    *  Loop over boxes to collapse entries of RAP when period = 1 in
    *  the coarsened direction.
    *-----------------------------------------------------------------*/

   if (hypre_IndexD(hypre_StructGridPeriodic(cgrid),cdir) == 1)
   {
      hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

            cstart = hypre_BoxIMin(cgrid_box);
            hypre_BoxGetSize(cgrid_box, loop_size);

            RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

            /*--------------------------------------------------------------
             * Computational loop. A loop over stored entries of RAP.
             *-------------------------------------------------------------*/
            for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
            {
               if (coarse_symm_elements[RAPloop] == -1)
               {
                  hypre_CopyIndex(coarse_stencil_shape[RAPloop], index);
                  switch (hypre_IndexD(index, cdir))
                  {
                     /*-----------------------------------------------------
                      * If RAP stencil index is 0 in coarsened direction,
                      * leave entry unchanged.
                      *-----------------------------------------------------*/
 
                      case 0:

                      break;

                     /*-----------------------------------------------------
                      * If RAP stencil index is +/-1 in coarsened direction,
                      * to add entry to cooresponding entry with 0 in the
                      * coarsened direction. Also zero out current index.
                      *-----------------------------------------------------*/
 
                      default:

                     /*---------------------------------------------------------
                      * Get pointer to the current RAP index (rap_ptrD)
                      * and cooresponding index with 0 in the coarsened
                      * direction (rap_ptrS).
                      *---------------------------------------------------------*/
                      rap_ptrD = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);
                      hypre_IndexD(index,cdir) = 0;
                      rap_ptrS = hypre_StructMatrixExtractPointerByIndex(RAP,
                                    ci, index);

                      /*--------------------------------------------------
                       * If RAP stencil index is zero except in coarsened
                       * direction and RAP is symmetric, must
                       * double entry when modifying the diagonal.
                       *--------------------------------------------------*/
                      symm_path_multiplier = 1;
                      diag = 0;
                      for (d = 0; d < dim; d++)
                      {
                         diag += hypre_IndexD(index,d) * hypre_IndexD(index,d);
                      }
                      if (diag == 0 && hypre_StructMatrixSymmetric(RAP))
                      {
                         symm_path_multiplier = 2;
                      }

                      hypre_BoxLoop1Begin(loop_size,
                                          RAP_dbox, cstart, stridec, iAc);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iAc
#include "hypre_box_smp_forloop.h"
                      hypre_BoxLoop1For(loopi, loopj, loopk, iAc)
                         {
                            rap_ptrS[iAc] += symm_path_multiplier *
                                             (rap_ptrD[iAc]);

                            rap_ptrD[iAc] = zero;
                         }
                      hypre_BoxLoop1End(iAc);

                      break;

                  } /* end of switch */

               } /* end if coarse_symm_element == -1 */

            } /* end of RAPloop */

         } /* end ForBoxI */

      } /* if periodic */

   return ierr;
}

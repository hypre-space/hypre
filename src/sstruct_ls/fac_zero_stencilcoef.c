/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "fac.h"

#define AbsStencilShape(stencil, abs_shape) \
{\
   HYPRE_Int ii,jj,kk;\
   ii = hypre_IndexX(stencil);\
   jj = hypre_IndexY(stencil);\
   kk = hypre_IndexZ(stencil);\
   abs_shape= abs(ii) + abs(jj) + abs(kk); \
}

/*--------------------------------------------------------------------------
 * hypre_FacZeroCFSten: Zeroes the coarse stencil coefficients that reach 
 * into an underlying coarsened refinement box.
 * Algo: For each cbox
 *       {
 *          1) refine cbox and expand by one in each direction
 *          2) boxman_intersect with the fboxman 
 *                3) loop over intersection boxes to see if stencil
 *                   reaches over.
 *       }
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FacZeroCFSten( hypre_SStructPMatrix *Af,
                     hypre_SStructPMatrix *Ac,
                     hypre_SStructGrid    *grid,
                     HYPRE_Int             fine_part,
                     hypre_Index           rfactors )
{
   hypre_BoxManager      *fboxman;
   hypre_BoxManEntry    **boxman_entries;
   HYPRE_Int              nboxman_entries;

   hypre_SStructPGrid    *p_cgrid;

   hypre_Box              fgrid_box;
   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;
   hypre_Box              scaled_box;

   hypre_Box             *shift_ibox;

   hypre_StructMatrix    *smatrix;

   hypre_StructStencil   *stencils;
   HYPRE_Int              stencil_size;

   hypre_Index            refine_factors, upper_shift;
   hypre_Index            stride;
   hypre_Index            stencil_shape;
   hypre_Index            zero_index, ilower, iupper;

   HYPRE_Int              nvars, var1, var2;
   HYPRE_Int              ndim;

   hypre_Box             *ac_dbox;
   double                *ac_ptr;
   hypre_Index            loop_size;

   HYPRE_Int              loopi, loopj, loopk, iac;
   HYPRE_Int              ci, i, j;

   HYPRE_Int              abs_shape;

   HYPRE_Int              ierr = 0;

   p_cgrid  = hypre_SStructPMatrixPGrid(Ac);
   nvars    = hypre_SStructPMatrixNVars(Ac);
   ndim     = hypre_SStructPGridNDim(p_cgrid);

   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(stride);
   hypre_ClearIndex(upper_shift);
   for (i= 0; i< ndim; i++)
   {
      stride[i]= 1;
      upper_shift[i]= rfactors[i]-1;
   }

   hypre_CopyIndex(rfactors, refine_factors);
   if (ndim < 3)
   {
      for (i= ndim; i< 3; i++)
      {
         refine_factors[i]= 1;
      }
   }
        
   for (var1= 0; var1< nvars; var1++)
   {
      cgrid= hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(Ac), var1);
      cgrid_boxes= hypre_StructGridBoxes(cgrid);

      fboxman= hypre_SStructGridBoxManager(grid, fine_part, var1);

     /*------------------------------------------------------------------
      * For each parent coarse box find all fboxes that may be connected
      * through a stencil entry- refine this box, expand it by one
      * in each direction, and boxman_intersect with fboxman
      *------------------------------------------------------------------*/
      hypre_ForBoxI(ci, cgrid_boxes)
      {
          cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

          hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), zero_index,
                                      refine_factors, hypre_BoxIMin(&scaled_box));
          hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), upper_shift,
                                      refine_factors, hypre_BoxIMax(&scaled_box));

          hypre_SubtractIndex(hypre_BoxIMin(&scaled_box), stride,
                              hypre_BoxIMin(&scaled_box));
          hypre_AddIndex(hypre_BoxIMax(&scaled_box), stride,
                         hypre_BoxIMax(&scaled_box));

          hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&scaled_box),
                                hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

          for (var2= 0; var2< nvars; var2++)
          {
             stencils=  hypre_SStructPMatrixSStencil(Ac, var1, var2);

             if (stencils != NULL)
             {
                stencil_size= hypre_StructStencilSize(stencils);
                smatrix     = hypre_SStructPMatrixSMatrix(Ac, var1, var2);
                ac_dbox     = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix),
                                                ci);

               /*---------------------------------------------------------
                * Find the stencil coefficients that must be zeroed off.
                * Loop over all possible boxes.
                *---------------------------------------------------------*/
                for (i= 0; i< stencil_size; i++)
                {
                   hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                   stencil_shape);
                   AbsStencilShape(stencil_shape, abs_shape);         

                   if (abs_shape)   /* non-centre stencils are zeroed */
                   {
                     /* look for connecting fboxes that must be zeroed. */
                      for (j= 0; j< nboxman_entries; j++)
                      {
                         hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
                         hypre_BoxSetExtents(&fgrid_box, ilower, iupper);

                         shift_ibox= hypre_CF_StenBox(&fgrid_box, cgrid_box, stencil_shape, 
                                                       refine_factors, ndim);

                         if ( hypre_BoxVolume(shift_ibox) )
                         {
                            ac_ptr= hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                            ci,
                                                                            stencil_shape);
                            hypre_BoxGetSize(shift_ibox, loop_size);

                            hypre_BoxLoop1Begin(loop_size, ac_dbox,
                                                hypre_BoxIMin(shift_ibox),
                                                stride, iac);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iac
#include "hypre_box_smp_forloop.h"
                            hypre_BoxLoop1For(loopi, loopj, loopk, iac)
                            {
                                ac_ptr[iac] = 0.0;
                            }
                            hypre_BoxLoop1End(iac);
                         }   /* if ( hypre_BoxVolume(shift_ibox) ) */

                         hypre_BoxDestroy(shift_ibox);

                      }  /* for (j= 0; j< nboxman_entries; j++) */
                   }     /* if (abs_shape)  */
                }        /* for (i= 0; i< stencil_size; i++) */
             }           /* if (stencils != NULL) */
          }              /* for (var2= 0; var2< nvars; var2++) */

          hypre_TFree(boxman_entries);
      }   /* hypre_ForBoxI  ci */
   }      /* for (var1= 0; var1< nvars; var1++) */

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FacZeroFCSten: Zeroes the fine stencil coefficients that reach
 * into a coarse box.
 * Idea: zero off any stencil connection of a fine box that does not
 *       connect to a sibling box
 * Algo: For each fbox
 *       {
 *          1) expand by one in each direction so that sibling boxes can be
 *             reached
 *          2) boxman_intersect with the fboxman to get all fboxes including
 *             itself and the siblings
 *          3) loop over intersection boxes, shift them in the stencil
 *             direction (now we are off the fbox), and subtract any sibling 
 *             extents. The remaining chunks (boxes of a box_array) are
 *             the desired but shifted extents.
 *          4) shift these shifted extents in the negative stencil direction
 *             to get back into fbox. Zero-off the matrix over these latter
 *             extents.
 *       }
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FacZeroFCSten( hypre_SStructPMatrix  *A,
                     hypre_SStructGrid     *grid,
                     HYPRE_Int              fine_part)
{
   MPI_Comm               comm=   hypre_SStructGridComm(grid); 
   hypre_BoxManager      *fboxman;
   hypre_BoxManEntry    **boxman_entries;
   HYPRE_Int              nboxman_entries;

   hypre_SStructPGrid    *p_fgrid;
   hypre_StructGrid      *fgrid;
   hypre_BoxArray        *fgrid_boxes;
   hypre_Box             *fgrid_box;
   hypre_Box              scaled_box;


   hypre_BoxArray        *intersect_boxes, *tmp_box_array1, *tmp_box_array2;

   hypre_StructMatrix    *smatrix;

   hypre_StructStencil   *stencils;
   HYPRE_Int              stencil_size;

   hypre_Index            stride, ilower, iupper;
   hypre_Index            stencil_shape, shift_index;

   hypre_Box              shift_ibox;
   hypre_Box              intersect_box;
   hypre_Index            size_ibox;

   HYPRE_Int              nvars, var1, var2;
   HYPRE_Int              ndim;

   hypre_Box             *a_dbox;
   double                *a_ptr;
   hypre_Index            loop_size;

   HYPRE_Int              loopi, loopj, loopk, ia;
   HYPRE_Int              fi, fj, i, j;
   HYPRE_Int              abs_shape;
   HYPRE_Int              myid, proc;
   HYPRE_Int              ierr = 0;

   hypre_MPI_Comm_rank(comm, &myid);

   p_fgrid  = hypre_SStructPMatrixPGrid(A);
   nvars    = hypre_SStructPMatrixNVars(A);
   ndim     = hypre_SStructPGridNDim(p_fgrid);

   hypre_ClearIndex(stride);
   for (i= 0; i< ndim; i++)
   {
      stride[i]= 1;
   }

   tmp_box_array1= hypre_BoxArrayCreate(1);

   for (var1= 0; var1< nvars; var1++)
   {
      fgrid      = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A), var1);
      fgrid_boxes= hypre_StructGridBoxes(fgrid);
      fboxman    = hypre_SStructGridBoxManager(grid, fine_part, var1);

      hypre_ForBoxI(fi, fgrid_boxes)
      {
         fgrid_box= hypre_BoxArrayBox(fgrid_boxes, fi);
         hypre_ClearIndex(size_ibox);
         for (i= 0; i< ndim; i++)
         {
            size_ibox[i] = hypre_BoxSizeD(fgrid_box, i) - 1;
         }

        /* expand fgrid_box & boxman_intersect with fboxman. */
         hypre_SubtractIndex(hypre_BoxIMin(fgrid_box), stride,
                             hypre_BoxIMin(&scaled_box));
         hypre_AddIndex(hypre_BoxIMax(fgrid_box), stride,
                        hypre_BoxIMax(&scaled_box));

         hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);
         
         for (var2= 0; var2< nvars; var2++)
         {
            stencils=  hypre_SStructPMatrixSStencil(A, var1, var2);

            if (stencils != NULL)
            {
               stencil_size= hypre_StructStencilSize(stencils);
               smatrix     = hypre_SStructPMatrixSMatrix(A, var1, var2);
               a_dbox      = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix),
                                               fi);

               for (i= 0; i< stencil_size; i++)
               {
                   hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                   stencil_shape);
                   AbsStencilShape(stencil_shape, abs_shape);

                   if (abs_shape)   /* non-centre stencils are zeroed */
                   {
                      hypre_SetIndex(shift_index,
                                     size_ibox[0]*stencil_shape[0],
                                     size_ibox[1]*stencil_shape[1],
                                     size_ibox[2]*stencil_shape[2]);
                      hypre_AddIndex(shift_index, hypre_BoxIMin(fgrid_box),
                                     hypre_BoxIMin(&shift_ibox));
                      hypre_AddIndex(shift_index, hypre_BoxIMax(fgrid_box),
                                     hypre_BoxIMax(&shift_ibox));
                      hypre_IntersectBoxes(&shift_ibox, fgrid_box, &shift_ibox);

                      hypre_SetIndex(shift_index, -stencil_shape[0], -stencil_shape[1],
                                    -stencil_shape[2]);

                      /*-----------------------------------------------------------
                       * Check to see if the stencil does not couple to a sibling
                       * box. These boxes should be in boxman_entries. But do not
                       * subtract fgrid_box itself, which is also in boxman_entries.
                       *-----------------------------------------------------------*/
                      hypre_AddIndex(stencil_shape, hypre_BoxIMin(&shift_ibox),
                                     hypre_BoxIMin(&shift_ibox));
                      hypre_AddIndex(stencil_shape, hypre_BoxIMax(&shift_ibox),
                                     hypre_BoxIMax(&shift_ibox));

                      intersect_boxes=  hypre_BoxArrayCreate(1);
                      hypre_CopyBox(&shift_ibox, hypre_BoxArrayBox(intersect_boxes,0));
 
                      for (j= 0; j< nboxman_entries; j++)
                      {
                         hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
                         hypre_SStructBoxManEntryGetBoxnum(boxman_entries[j], &fj);

                         if ((proc != myid) || (fj != fi))
                         {
                            hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
                            hypre_BoxSetExtents(&scaled_box, ilower, iupper);

                            hypre_IntersectBoxes(&shift_ibox, &scaled_box, &intersect_box);

                            if ( hypre_BoxVolume(&intersect_box) )
                            {
                               hypre_CopyBox(&intersect_box,
                                              hypre_BoxArrayBox(tmp_box_array1, 0));

                               tmp_box_array2= hypre_BoxArrayCreate(0);

                               hypre_SubtractBoxArrays(intersect_boxes,
                                                       tmp_box_array1,
                                                       tmp_box_array2);

                               hypre_BoxArrayDestroy(tmp_box_array2);
                            }
                         }
                      }   /* for (j= 0; j< nboxman_entries; j++) */

                     /*-----------------------------------------------------------
                      * intersect_boxes now has the shifted extents for the
                      * coefficients to be zeroed.
                      *-----------------------------------------------------------*/
                      a_ptr= hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                     fi,
                                                                     stencil_shape);
                      hypre_ForBoxI(fj, intersect_boxes)
                      {
                         hypre_CopyBox(hypre_BoxArrayBox(intersect_boxes, fj), &intersect_box);

                         hypre_AddIndex(shift_index, hypre_BoxIMin(&intersect_box),
                                        hypre_BoxIMin(&intersect_box));
                         hypre_AddIndex(shift_index, hypre_BoxIMax(&intersect_box),
                                        hypre_BoxIMax(&intersect_box));

                         hypre_BoxGetSize(&intersect_box, loop_size);

                         hypre_BoxLoop1Begin(loop_size, a_dbox,
                                             hypre_BoxIMin(&intersect_box),
                                             stride, ia);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,ia
#include "hypre_box_smp_forloop.h"
                         hypre_BoxLoop1For(loopi, loopj, loopk, ia)
                         {
                             a_ptr[ia] = 0.0;
                         }
                         hypre_BoxLoop1End(ia);

                      }  /* hypre_ForBoxI(fj, intersect_boxes) */

                      hypre_BoxArrayDestroy(intersect_boxes);

                   }  /* if (abs_shape) */
               }      /* for (i= 0; i< stencil_size; i++) */
            }         /* if (stencils != NULL) */
         }            /* for (var2= 0; var2< nvars; var2++) */

         hypre_TFree(boxman_entries);
      }  /* hypre_ForBoxI(fi, fgrid_boxes) */
   }     /* for (var1= 0; var1< nvars; var1++) */

   hypre_BoxArrayDestroy(tmp_box_array1);

   return ierr;
}


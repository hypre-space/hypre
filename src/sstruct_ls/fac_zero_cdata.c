/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/




#include "headers.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_FacZeroCData: Zeroes the data over the underlying coarse indices of
 * the refinement patches.
 *    Algo.:For each cbox
 *       {
 *          1) refine cbox and boxman_intersect with fboxman
 *          2) loop over intersection boxes
 *                3) coarsen and contract (only the coarse nodes on this 
 *                   processor) and zero data.
 *       }
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FacZeroCData( void                 *fac_vdata,
                    hypre_SStructMatrix  *A )
{
   hypre_FACData         *fac_data      =  fac_vdata;

   hypre_SStructGrid     *grid;
   hypre_SStructPGrid    *p_cgrid;

   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;

   hypre_BoxManager      *fboxman;
   hypre_BoxManEntry    **boxman_entries;
   HYPRE_Int              nboxman_entries;

   hypre_Box              scaled_box;
   hypre_Box              intersect_box;

   hypre_SStructPMatrix  *level_pmatrix;
   hypre_StructStencil   *stencils;
   HYPRE_Int              stencil_size;

   hypre_Index           *refine_factors;
   hypre_Index            temp_index;
   hypre_Index            ilower, iupper;
  
   HYPRE_Int              max_level     =  fac_data -> max_levels;
   HYPRE_Int             *level_to_part =  fac_data -> level_to_part;

   HYPRE_Int              ndim          =  hypre_SStructMatrixNDim(A);
   HYPRE_Int              part_crse     =  0;
   HYPRE_Int              part_fine     =  1;
   HYPRE_Int              level;
   HYPRE_Int              nvars, var;

   HYPRE_Int              ci, i, j, rem, intersect_size;

   double                *values;
 
   HYPRE_Int              ierr = 0;

   for (level= max_level; level> 0; level--)
   {
      level_pmatrix = hypre_SStructMatrixPMatrix(fac_data -> A_level[level], part_crse);

      grid          = (fac_data -> grid_level[level]);
      refine_factors= &(fac_data -> refine_factors[level]);
      
      p_cgrid= hypre_SStructGridPGrid(grid, part_crse);
      nvars  = hypre_SStructPGridNVars(p_cgrid);

      for (var= 0; var< nvars; var++)
      {
         stencils    =  hypre_SStructPMatrixSStencil(level_pmatrix, var, var);
         stencil_size=  hypre_StructStencilSize(stencils);

         /*---------------------------------------------------------------------
          * For each variable, find the underlying boxes for each coarse box.
          *---------------------------------------------------------------------*/
         cgrid        = hypre_SStructPGridSGrid(p_cgrid, var);
         cgrid_boxes  = hypre_StructGridBoxes(cgrid);
         fboxman         = hypre_SStructGridBoxManager(grid, part_fine, var);

         hypre_ForBoxI(ci, cgrid_boxes)
         {
             cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

             hypre_ClearIndex(temp_index);
             hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), temp_index,
                                        *refine_factors, hypre_BoxIMin(&scaled_box));
             for (i= 0; i< ndim; i++)
             {
                temp_index[i]= (*refine_factors)[i]-1;
             }
             hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), temp_index,
                                        *refine_factors, hypre_BoxIMax(&scaled_box));

             hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&scaled_box),
                                   hypre_BoxIMax(&scaled_box), &boxman_entries,
                                  &nboxman_entries);

             for (i= 0; i< nboxman_entries; i++)
             {
                hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
                hypre_BoxSetExtents(&intersect_box, ilower, iupper);
                hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

               /* adjust the box so that it is divisible by refine_factors */
                for (j= 0; j< ndim; j++)
                {
                   rem= hypre_BoxIMin(&intersect_box)[j]%(*refine_factors)[j];
                   if (rem)
                   {
                      hypre_BoxIMin(&intersect_box)[j]+=(*refine_factors)[j] - rem;
                   }
                }

                hypre_ClearIndex(temp_index);
                hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect_box), temp_index,
                                           *refine_factors, hypre_BoxIMin(&intersect_box));
                hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect_box), temp_index,
                                           *refine_factors, hypre_BoxIMax(&intersect_box));

                intersect_size= hypre_BoxVolume(&intersect_box);
                if (intersect_size > 0)
                {
                  /*------------------------------------------------------------
                   * Coarse underlying box found. Now zero off.
                   *------------------------------------------------------------*/
                   values= hypre_CTAlloc(double, intersect_size);

                   for (j= 0; j< stencil_size; j++)
                   {
                      HYPRE_SStructMatrixSetBoxValues(fac_data -> A_level[level],
                                                      part_crse, 
                                                      hypre_BoxIMin(&intersect_box),
                                                      hypre_BoxIMax(&intersect_box),
                                                      var, 1, &j, values);

                      HYPRE_SStructMatrixSetBoxValues(A,
                                                      level_to_part[level-1], 
                                                      hypre_BoxIMin(&intersect_box),
                                                      hypre_BoxIMax(&intersect_box),
                                                      var, 1, &j, values);
                   }

                   hypre_TFree(values);

                }  /* if (intersect_size > 0) */
             }     /* for (i= 0; i< nboxman_entries; i++) */

             hypre_TFree(boxman_entries);

         }   /* hypre_ForBoxI(ci, cgrid_boxes) */
      }      /* for (var= 0; var< nvars; var++) */
   }         /* for (level= max_level; level> 0; level--) */

   return ierr;
}


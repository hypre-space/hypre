/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_ZeroAMRVectorData: Zeroes the data over the underlying coarse
 * indices of the refinement patches.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ZeroAMRVectorData(hypre_SStructVector  *b,
                        HYPRE_Int            *plevels,
                        hypre_Index          *rfactors )
{
   hypre_SStructGrid     *grid =  hypre_SStructVectorGrid(b);
   hypre_SStructPGrid    *p_cgrid;

   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;

   hypre_BoxManager      *fboxman;
   hypre_BoxManEntry    **boxman_entries;
   HYPRE_Int              nboxman_entries;

   hypre_Box              scaled_box;
   hypre_Box              intersect_box;

   HYPRE_Int              npart =  hypre_SStructVectorNParts(b);
   HYPRE_Int              ndim =  hypre_SStructVectorNDim(b);

   HYPRE_Int             *levels;

   hypre_Index           *refine_factors;
   hypre_Index            temp_index, ilower, iupper;

   HYPRE_Int              level;
   HYPRE_Int              nvars, var;

   HYPRE_Int              part, ci, rem, i, j, intersect_size;

   HYPRE_Real            *values1;

   HYPRE_Int              ierr = 0;

   hypre_BoxInit(&scaled_box, ndim);
   hypre_BoxInit(&intersect_box, ndim);

   levels        = hypre_CTAlloc(HYPRE_Int,  npart, HYPRE_MEMORY_HOST);
   refine_factors = hypre_CTAlloc(hypre_Index,  npart, HYPRE_MEMORY_HOST);
   for (part = 0; part < npart; part++)
   {
      levels[plevels[part]] = part;
      for (i = 0; i < ndim; i++)
      {
         refine_factors[plevels[part]][i] = rfactors[part][i];
      }
      for (i = ndim; i < 3; i++)
      {
         refine_factors[plevels[part]][i] = 1;
      }
   }

   hypre_ClearIndex(temp_index);

   for (level = npart - 1; level > 0; level--)
   {
      p_cgrid = hypre_SStructGridPGrid(grid, levels[level - 1]);
      nvars  = hypre_SStructPGridNVars(p_cgrid);

      for (var = 0; var < nvars; var++)
      {
         /*---------------------------------------------------------------------
          * For each variable, find the underlying boxes for each fine box.
          *---------------------------------------------------------------------*/
         cgrid      = hypre_SStructPGridSGrid(p_cgrid, var);
         cgrid_boxes = hypre_StructGridBoxes(cgrid);
         fboxman    = hypre_SStructGridBoxManager(grid, levels[level], var);

         hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

            hypre_ClearIndex(temp_index);
            hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), temp_index,
                                        refine_factors[level], hypre_BoxIMin(&scaled_box));
            for (i = 0; i < ndim; i++)
            {
               temp_index[i] = refine_factors[level][i] - 1;
            }
            hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), temp_index,
                                        refine_factors[level], hypre_BoxIMax(&scaled_box));
            hypre_ClearIndex(temp_index);

            hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&scaled_box),
                                  hypre_BoxIMax(&scaled_box), &boxman_entries,
                                  &nboxman_entries);

            for (i = 0; i < nboxman_entries; i++)
            {
               hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
               hypre_BoxSetExtents(&intersect_box, ilower, iupper);
               hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

               /* adjust the box so that it is divisible by refine_factors */
               for (j = 0; j < ndim; j++)
               {
                  rem = hypre_BoxIMin(&intersect_box)[j] % refine_factors[level][j];
                  if (rem)
                  {
                     hypre_BoxIMin(&intersect_box)[j] += refine_factors[level][j] - rem;
                  }
               }

               hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect_box), temp_index,
                                           refine_factors[level], hypre_BoxIMin(&intersect_box));
               hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect_box), temp_index,
                                           refine_factors[level], hypre_BoxIMax(&intersect_box));

               intersect_size = hypre_BoxVolume(&intersect_box);
               if (intersect_size > 0)
               {
                  /*------------------------------------------------------------
                   * Coarse underlying box found. Now zero off.
                   *------------------------------------------------------------*/
                  values1 = hypre_CTAlloc(HYPRE_Real,  intersect_size, HYPRE_MEMORY_HOST);

                  HYPRE_SStructVectorSetBoxValues(b, levels[level - 1],
                                                  hypre_BoxIMin(&intersect_box),
                                                  hypre_BoxIMax(&intersect_box),
                                                  var, values1);
                  hypre_TFree(values1, HYPRE_MEMORY_HOST);

               }  /* if (intersect_size > 0) */
            }     /* for (i= 0; i< nboxman_entries; i++) */

            hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

         }   /* hypre_ForBoxI(ci, cgrid_boxes) */
      }      /* for (var= 0; var< nvars; var++) */
   }         /* for (level= max_level; level> 0; level--) */

   hypre_TFree(levels, HYPRE_MEMORY_HOST);
   hypre_TFree(refine_factors, HYPRE_MEMORY_HOST);

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_ZeroAMRMatrixData: Zeroes the data over the underlying coarse
 * indices of the refinement patches between two levels.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ZeroAMRMatrixData(hypre_SStructMatrix  *A,
                        HYPRE_Int             part_crse,
                        hypre_Index           rfactors )
{
   hypre_SStructGraph    *graph =  hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid =  hypre_SStructGraphGrid(graph);
   HYPRE_Int              ndim =  hypre_SStructMatrixNDim(A);

   hypre_SStructPGrid    *p_cgrid;

   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;

   hypre_BoxManager      *fboxman;
   hypre_BoxManEntry    **boxman_entries;
   HYPRE_Int              nboxman_entries;

   hypre_Box              scaled_box;
   hypre_Box              intersect_box;

   hypre_SStructStencil  *stencils;
   HYPRE_Int              stencil_size;

   hypre_Index           *stencil_shape;
   hypre_Index            temp_index, ilower, iupper;

   HYPRE_Int              nvars, var;

   HYPRE_Int              ci, i, j, rem, intersect_size, rank;

   HYPRE_Real            *values1, *values2;

   HYPRE_Int              ierr = 0;

   hypre_BoxInit(&scaled_box, ndim);
   hypre_BoxInit(&intersect_box, ndim);

   p_cgrid = hypre_SStructGridPGrid(grid, part_crse);
   nvars  = hypre_SStructPGridNVars(p_cgrid);

   for (var = 0; var < nvars; var++)
   {
      stencils     =  hypre_SStructGraphStencil(graph, part_crse, var);
      stencil_size =  hypre_SStructStencilSize(stencils);
      stencil_shape = hypre_SStructStencilShape(stencils);

      /*---------------------------------------------------------------------
       * For each variable, find the underlying boxes for each fine box.
       *---------------------------------------------------------------------*/
      cgrid        = hypre_SStructPGridSGrid(p_cgrid, var);
      cgrid_boxes  = hypre_StructGridBoxes(cgrid);
      fboxman      = hypre_SStructGridBoxManager(grid, part_crse + 1, var);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         hypre_ClearIndex(temp_index);
         hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), temp_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         for (i = 0; i < ndim; i++)
         {
            temp_index[i] =  rfactors[i] - 1;
         }
         hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), temp_index,
                                     rfactors, hypre_BoxIMax(&scaled_box));
         hypre_ClearIndex(temp_index);

         hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&intersect_box, ilower, iupper);
            hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

            /* adjust the box so that it is divisible by refine_factors */
            for (j = 0; j < ndim; j++)
            {
               rem = hypre_BoxIMin(&intersect_box)[j] % rfactors[j];
               if (rem)
               {
                  hypre_BoxIMin(&intersect_box)[j] += rfactors[j] - rem;
               }
            }

            hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect_box), temp_index,
                                        rfactors, hypre_BoxIMin(&intersect_box));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect_box), temp_index,
                                        rfactors, hypre_BoxIMax(&intersect_box));

            intersect_size = hypre_BoxVolume(&intersect_box);
            if (intersect_size > 0)
            {
               /*------------------------------------------------------------
                * Coarse underlying box found. Now zero off.
                *------------------------------------------------------------*/
               values1 = hypre_CTAlloc(HYPRE_Real,  intersect_size, HYPRE_MEMORY_HOST);
               values2 = hypre_TAlloc(HYPRE_Real,  intersect_size, HYPRE_MEMORY_HOST);
               for (j = 0; j < intersect_size; j++)
               {
                  values2[j] = 1.0;
               }

               for (j = 0; j < stencil_size; j++)
               {
                  rank = hypre_abs(hypre_IndexX(stencil_shape[j])) +
                         hypre_abs(hypre_IndexY(stencil_shape[j])) +
                         hypre_abs(hypre_IndexZ(stencil_shape[j]));

                  if (rank)
                  {
                     HYPRE_SStructMatrixSetBoxValues(A,
                                                     part_crse,
                                                     hypre_BoxIMin(&intersect_box),
                                                     hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values1);
                  }
                  else
                  {
                     HYPRE_SStructMatrixSetBoxValues(A,
                                                     part_crse,
                                                     hypre_BoxIMin(&intersect_box),
                                                     hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values2);
                  }
               }
               hypre_TFree(values1, HYPRE_MEMORY_HOST);
               hypre_TFree(values2, HYPRE_MEMORY_HOST);

            }   /* if (intersect_size > 0) */
         }      /* for (i= 0; i< nmap_entries; i++) */

         hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);
      }   /* hypre_ForBoxI(ci, cgrid_boxes) */
   }      /* for (var= 0; var< nvars; var++) */

   return ierr;
}




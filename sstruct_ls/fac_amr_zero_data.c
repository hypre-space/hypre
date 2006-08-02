/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ZeroAMRVectorData: Zeroes the data over the underlying coarse 
 * indices of the refinement patches.
 *--------------------------------------------------------------------------*/

int
hypre_ZeroAMRVectorData(hypre_SStructVector  *b,
                        int                  *plevels,
                        hypre_Index          *rfactors )
{
   hypre_SStructGrid     *grid =  hypre_SStructVectorGrid(b); 
   hypre_SStructPGrid    *p_cgrid;

   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;

   hypre_BoxMap          *fmap;
   hypre_BoxMapEntry    **map_entries;
   int                    nmap_entries;

   hypre_Box              scaled_box;
   hypre_Box              intersect_box;

   int                    npart=  hypre_SStructVectorNParts(b);
   int                    ndim =  hypre_SStructVectorNDim(b);

   int                   *levels;

   hypre_Index           *refine_factors;
   hypre_Index            temp_index, ilower, iupper;
  
   int                    level;
   int                    nvars, var;

   int                    part, ci, rem, i, j, intersect_size;

   double                *values1;
 
   int                    ierr = 0;

   levels        = hypre_CTAlloc(int, npart);
   refine_factors= hypre_CTAlloc(hypre_Index, npart);
   for (part= 0; part< npart; part++)
   {
       levels[plevels[part]]= part;
       for (i= 0; i< ndim; i++)
       {
           refine_factors[plevels[part]][i]= rfactors[part][i];
       }
       for (i= ndim; i< 3; i++)
       {
           refine_factors[plevels[part]][i]= 1;
       }
   }

   hypre_ClearIndex(temp_index);

   for (level= npart-1; level> 0; level--)
   {
      p_cgrid= hypre_SStructGridPGrid(grid, levels[level-1]);
      nvars  = hypre_SStructPGridNVars(p_cgrid);

      for (var= 0; var< nvars; var++)
      {
         /*---------------------------------------------------------------------
          * For each variable, find the underlying boxes for each fine box.
          *---------------------------------------------------------------------*/
         cgrid      = hypre_SStructPGridSGrid(p_cgrid, var);
         cgrid_boxes= hypre_StructGridBoxes(cgrid);
         fmap       = hypre_SStructGridMap(grid, levels[level], var);

         hypre_ForBoxI(ci, cgrid_boxes)
         {
             cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

             hypre_ClearIndex(temp_index);
             hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), temp_index,
                            refine_factors[level], hypre_BoxIMin(&scaled_box));
             for (i= 0; i< ndim; i++)
             {
                temp_index[i]= refine_factors[level][i]-1;
             }
             hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), temp_index,
                            refine_factors[level], hypre_BoxIMax(&scaled_box));
             hypre_ClearIndex(temp_index);

             hypre_BoxMapIntersect(fmap, hypre_BoxIMin(&scaled_box),
                                   hypre_BoxIMax(&scaled_box), &map_entries,
                                  &nmap_entries);

             for (i= 0; i< nmap_entries; i++)
             {
                hypre_BoxMapEntryGetExtents(map_entries[i], ilower, iupper);
                hypre_BoxSetExtents(&intersect_box, ilower, iupper);
                hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

               /* adjust the box so that it is divisible by refine_factors */
                for (j= 0; j< ndim; j++)
                {
                   rem= hypre_BoxIMin(&intersect_box)[j]%refine_factors[level][j];
                   if (rem)
                   {
                      hypre_BoxIMin(&intersect_box)[j]+=refine_factors[level][j] - rem;
                   }
                }

                hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect_box), temp_index,
                                     refine_factors[level], hypre_BoxIMin(&intersect_box));
                hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect_box), temp_index,
                                     refine_factors[level], hypre_BoxIMax(&intersect_box));

                intersect_size= hypre_BoxVolume(&intersect_box);
                if (intersect_size > 0)
                {
                  /*------------------------------------------------------------
                   * Coarse underlying box found. Now zero off.
                   *------------------------------------------------------------*/
                   values1= hypre_CTAlloc(double, intersect_size);

                   HYPRE_SStructVectorSetBoxValues(b, levels[level-1], 
                                                   hypre_BoxIMin(&intersect_box),
                                                   hypre_BoxIMax(&intersect_box),
                                                   var, values1);
                   hypre_TFree(values1);

                }  /* if (intersect_size > 0) */
             }     /* for (i= 0; i< nmap_entries; i++) */
       
             hypre_TFree(map_entries);

         }   /* hypre_ForBoxI(ci, cgrid_boxes) */
      }      /* for (var= 0; var< nvars; var++) */
   }         /* for (level= max_level; level> 0; level--) */

   hypre_TFree(levels);
   hypre_TFree(refine_factors);

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_ZeroAMRMatrixData: Zeroes the data over the underlying coarse 
 * indices of the refinement patches between two levels.
 *--------------------------------------------------------------------------*/

int
hypre_ZeroAMRMatrixData(hypre_SStructMatrix  *A,
                        int                   part_crse,
                        hypre_Index           rfactors )
{
   hypre_SStructGraph    *graph=  hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid =  hypre_SStructGraphGrid(graph); 
   int                    ndim =  hypre_SStructMatrixNDim(A);

   hypre_SStructPGrid    *p_cgrid;

   hypre_StructGrid      *cgrid;
   hypre_BoxArray        *cgrid_boxes;
   hypre_Box             *cgrid_box;

   hypre_BoxMap          *fmap;
   hypre_BoxMapEntry    **map_entries;
   int                    nmap_entries;

   hypre_Box              scaled_box;
   hypre_Box              intersect_box;

   hypre_SStructStencil  *stencils;
   int                    stencil_size;

   hypre_Index           *stencil_shape;
   hypre_Index            temp_index, ilower, iupper;
  
   int                    nvars, var;

   int                    ci, i, j, rem, intersect_size, rank;

   double                *values1, *values2;
 
   int                    ierr = 0;

   p_cgrid= hypre_SStructGridPGrid(grid, part_crse);
   nvars  = hypre_SStructPGridNVars(p_cgrid);

   for (var= 0; var< nvars; var++)
   {
      stencils     =  hypre_SStructGraphStencil(graph, part_crse, var);
      stencil_size =  hypre_SStructStencilSize(stencils);
      stencil_shape= hypre_SStructStencilShape(stencils);

      /*---------------------------------------------------------------------
       * For each variable, find the underlying boxes for each fine box.
       *---------------------------------------------------------------------*/
      cgrid        = hypre_SStructPGridSGrid(p_cgrid, var);
      cgrid_boxes  = hypre_StructGridBoxes(cgrid);
      fmap         = hypre_SStructGridMap(grid, part_crse+1, var);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
          cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

          hypre_ClearIndex(temp_index);
          hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), temp_index,
                                      rfactors, hypre_BoxIMin(&scaled_box));
          for (i= 0; i< ndim; i++)
          {
             temp_index[i]=  rfactors[i]-1;
          }
          hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), temp_index,
                                      rfactors, hypre_BoxIMax(&scaled_box));
          hypre_ClearIndex(temp_index);

          hypre_BoxMapIntersect(fmap, hypre_BoxIMin(&scaled_box),
                                hypre_BoxIMax(&scaled_box), &map_entries,
                               &nmap_entries);

          for (i= 0; i< nmap_entries; i++)
          {
             hypre_BoxMapEntryGetExtents(map_entries[i], ilower, iupper);
             hypre_BoxSetExtents(&intersect_box, ilower, iupper);
             hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

            /* adjust the box so that it is divisible by refine_factors */
             for (j= 0; j< ndim; j++)
             {
                rem= hypre_BoxIMin(&intersect_box)[j]%rfactors[j];
                if (rem)
                {
                    hypre_BoxIMin(&intersect_box)[j]+= rfactors[j] - rem;
                }
             }

             hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect_box), temp_index,
                                         rfactors, hypre_BoxIMin(&intersect_box));
             hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect_box), temp_index,
                                         rfactors, hypre_BoxIMax(&intersect_box));

             intersect_size= hypre_BoxVolume(&intersect_box);
             if (intersect_size > 0)
             {
                /*------------------------------------------------------------
                 * Coarse underlying box found. Now zero off.
                 *------------------------------------------------------------*/
                 values1= hypre_CTAlloc(double, intersect_size);
                 values2= hypre_TAlloc(double, intersect_size);
                 for (j= 0; j< intersect_size; j++)
                 {
                     values2[j]= 1.0;
                 }

                 for (j= 0; j< stencil_size; j++)
                 {
                    rank= abs(hypre_IndexX(stencil_shape[j]))+
                          abs(hypre_IndexY(stencil_shape[j]))+
                          abs(hypre_IndexZ(stencil_shape[j]));
                   
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
                 hypre_TFree(values1);
                 hypre_TFree(values2);

             }   /* if (intersect_size > 0) */
          }      /* for (i= 0; i< nmap_entries; i++) */

          hypre_TFree(map_entries);
      }   /* hypre_ForBoxI(ci, cgrid_boxes) */
   }      /* for (var= 0; var< nvars; var++) */

   return ierr;
}




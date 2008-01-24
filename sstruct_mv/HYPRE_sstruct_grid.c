/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridCreate( MPI_Comm           comm,
                         int                ndim,
                         int                nparts,
                         HYPRE_SStructGrid *grid_ptr )
{
   hypre_SStructGrid       *grid;
   hypre_SStructPGrid     **pgrids;
   hypre_SStructPGrid      *pgrid;
   int                     *nneighbors;
   hypre_SStructNeighbor  **neighbors;
   int                      i;

   grid = hypre_TAlloc(hypre_SStructGrid, 1);

   hypre_SStructGridComm(grid)   = comm;
   hypre_SStructGridNDim(grid)   = ndim;
   hypre_SStructGridNParts(grid) = nparts;
   pgrids = hypre_TAlloc(hypre_SStructPGrid *, nparts);
   nneighbors = hypre_TAlloc(int, nparts);
   neighbors  = hypre_TAlloc(hypre_SStructNeighbor *, nparts);
   for (i = 0; i < nparts; i++)
   {
      hypre_SStructPGridCreate(comm, ndim, &pgrid);
      pgrids[i] = pgrid;
      nneighbors[i] = 0;
      neighbors[i]  = NULL;
   }
   hypre_SStructGridPGrids(grid)  = pgrids;
   hypre_SStructGridNNeighbors(grid) = nneighbors;
   hypre_SStructGridNeighbors(grid)  = neighbors;
   hypre_SStructGridNUCVars(grid) = 0;
   hypre_SStructGridUCVars(grid)  = NULL;

   hypre_SStructGridMaps(grid) = NULL;
   hypre_SStructGridInfo(grid) = NULL;

   /* miscellaneous */
   hypre_SStructGridLocalSize(grid)     = 0;
   hypre_SStructGridGlobalSize(grid)    = 0;
   hypre_SStructGridRefCount(grid)      = 1;

   /* GEC0902 ghost addition to the grid    */
   hypre_SStructGridGhlocalSize(grid)   = 0; 

   *grid_ptr = grid;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridDestroy( HYPRE_SStructGrid grid )
{
   int                        nparts;
   hypre_SStructPGrid       **pgrids;
   int                       *nneighbors;
   hypre_SStructNeighbor    **neighbors;
   int                      **nvneighbors;
   hypre_SStructNeighbor   ***vneighbors;
   hypre_SStructCommInfo    **vnbor_comm_info;
   int                        vnbor_ncomms;
   hypre_BoxMap            ***maps;
   hypre_SStructMapInfo    ***info;
   hypre_SStructNMapInfo   ***ninfo;
   int                        nvars;
   int                        part, var, i;

   if (grid)
   {
      hypre_SStructGridRefCount(grid) --;
      if (hypre_SStructGridRefCount(grid) == 0)
      {
         nparts  = hypre_SStructGridNParts(grid);
         pgrids  = hypre_SStructGridPGrids(grid);
         nneighbors  = hypre_SStructGridNNeighbors(grid);
         neighbors   = hypre_SStructGridNeighbors(grid);
         nvneighbors = hypre_SStructGridNVNeighbors(grid);
         vneighbors  = hypre_SStructGridVNeighbors(grid);
         vnbor_comm_info = hypre_SStructGridVNborCommInfo(grid);
         vnbor_ncomms = hypre_SStructGridVNborNComms(grid);
         maps  = hypre_SStructGridMaps(grid);
         info  = hypre_SStructGridInfo(grid);
         ninfo = hypre_SStructGridNInfo(grid);
         for (part = 0; part < nparts; part++)
         {
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               hypre_TFree(vneighbors[part][var]);
               hypre_BoxMapDestroy(maps[part][var]);
               hypre_TFree(info[part][var]);
               hypre_TFree(ninfo[part][var]);
            }
            hypre_TFree(neighbors[part]);
            hypre_TFree(nvneighbors[part]);
            hypre_TFree(vneighbors[part]);
            hypre_SStructPGridDestroy(pgrids[part]);
            hypre_TFree(maps[part]);
            hypre_TFree(info[part]);
            hypre_TFree(ninfo[part]);
         }
         for (i = 0; i < vnbor_ncomms; i++)
         {
            hypre_CommInfoDestroy(
               hypre_SStructCommInfoCommInfo(vnbor_comm_info[i]));
            hypre_TFree(vnbor_comm_info[i]);
         }
         hypre_TFree(vnbor_comm_info);
         hypre_TFree(pgrids);
         hypre_TFree(nneighbors);
         hypre_TFree(neighbors);
         hypre_TFree(nvneighbors);
         hypre_TFree(vneighbors);
         hypre_TFree(vnbor_comm_info);
         hypre_TFree(maps);
         hypre_TFree(info);
         hypre_TFree(ninfo);
         hypre_TFree(grid);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridSetExtents( HYPRE_SStructGrid  grid,
                             int                part,
                             int               *ilower,
                             int               *iupper )
{
   int                  ndim  = hypre_SStructGridNDim(grid);
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);
   hypre_Index          cilower;
   hypre_Index          ciupper;

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   hypre_SStructPGridSetExtents(pgrid, cilower, ciupper);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int HYPRE_SStructGridSetVariables( HYPRE_SStructGrid      grid,
                                   int                    part,
                                   int                    nvars,
                                   HYPRE_SStructVariable *vartypes )
{
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);

   hypre_SStructPGridSetVariables(pgrid, nvars, vartypes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridSetVariable
 * Like HYPRE_SStructGridSetVariables, but do one variable at a time.
 * Nevertheless, you still must provide nvars, for memory allocation.
 *--------------------------------------------------------------------------*/

int HYPRE_SStructGridSetVariable( HYPRE_SStructGrid      grid,
                                  int                    part,
                                  int                    var,
                                  int                    nvars,
                                  HYPRE_SStructVariable  vartype )
{
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);

   hypre_SStructPGridSetVariable( pgrid, var, nvars, vartype );

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridAddVariables( HYPRE_SStructGrid      grid,
                               int                    part,
                               int                   *index,
                               int                    nvars,
                               HYPRE_SStructVariable *vartypes )
{
   int                  ndim    = hypre_SStructGridNDim(grid);
   int                  nucvars = hypre_SStructGridNUCVars(grid);
   hypre_SStructUCVar **ucvars  = hypre_SStructGridUCVars(grid);
   hypre_SStructUCVar  *ucvar;

   int                  memchunk = 1000;
   int                  i;

   /* allocate more space if necessary */
   if ((nucvars % memchunk) == 0)
   {
      ucvars = hypre_TReAlloc(ucvars, hypre_SStructUCVar *,
                              (nucvars + memchunk));
   }

   ucvar = hypre_TAlloc(hypre_SStructUCVar, 1);
   hypre_SStructUCVarUVars(ucvar) = hypre_TAlloc(hypre_SStructUVar, nvars);
   hypre_SStructUCVarPart(ucvar) = part;
   hypre_CopyToCleanIndex(index, ndim, hypre_SStructUCVarCell(ucvar));
   hypre_SStructUCVarNUVars(ucvar) = nvars;
   for (i = 0; i < nvars; i++)
   {
      hypre_SStructUCVarType(ucvar, i) = vartypes[i];
      hypre_SStructUCVarRank(ucvar, i) = -1;           /* don't know, yet */
      hypre_SStructUCVarProc(ucvar, i) = -1;           /* don't know, yet */
   }
   ucvars[nucvars] = ucvar;
   nucvars++;

   hypre_SStructGridNUCVars(grid) = nucvars;
   hypre_SStructGridUCVars(grid)  = ucvars;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridAddVariable
 * Like HYPRE_SStructGridAddVariables, but just one variable at a time.
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridAddVariable( HYPRE_SStructGrid      grid,
                              int                    part,
                              int                   *index,
                              int                    var,
                              HYPRE_SStructVariable  vartype )
{
   int                  ndim    = hypre_SStructGridNDim(grid);
   int                  nucvars = hypre_SStructGridNUCVars(grid);
   hypre_SStructUCVar **ucvars  = hypre_SStructGridUCVars(grid);
   hypre_SStructUCVar  *ucvar;

   int                  memchunk = 1000;
   int                  nvars = 1;  /* each ucvar gets only one variable */

   /* allocate more space if necessary */
   if ((nucvars % memchunk) == 0)
   {
      ucvars = hypre_TReAlloc(ucvars, hypre_SStructUCVar *,
                              (nucvars + memchunk));
   }

   ucvar = hypre_TAlloc(hypre_SStructUCVar, 1);
   hypre_SStructUCVarUVars(ucvar) = hypre_TAlloc(hypre_SStructUVar, nvars);
   hypre_SStructUCVarPart(ucvar) = part;
   hypre_CopyToCleanIndex(index, ndim, hypre_SStructUCVarCell(ucvar));
   hypre_SStructUCVarNUVars(ucvar) = nvars;

   hypre_SStructUCVarType(ucvar, var) = vartype;
   hypre_SStructUCVarRank(ucvar, var) = -1;           /* don't know, yet */
   hypre_SStructUCVarProc(ucvar, var) = -1;           /* don't know, yet */

   ucvars[nucvars] = ucvar;
   nucvars++;

   hypre_SStructGridNUCVars(grid) = nucvars;
   hypre_SStructGridUCVars(grid)  = ucvars;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridSetNeighborPart( HYPRE_SStructGrid  grid,
                                  int                part,
                                  int               *ilower,
                                  int               *iupper,
                                  int                nbor_part,
                                  int               *nbor_ilower,
                                  int               *nbor_iupper,
                                  int               *index_map,
                                  int               *index_dir )
{
   int                      ndim       = hypre_SStructGridNDim(grid);
   int                     *nneighbors = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors  = hypre_SStructGridNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;

   hypre_Box               *box;
   hypre_Index              cilower;
   hypre_Index              ciupper;
   hypre_IndexRef           coord, dir, ilower_mapped;
   int                      memchunk = 10;
   int                      d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = hypre_TReAlloc(neighbors[part], hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk));
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nneighbors[part]++;

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxSetExtents(box, cilower, ciupper);

   hypre_SStructNeighborPart(neighbor) = nbor_part;

   coord = hypre_SStructNeighborCoord(neighbor);
   dir = hypre_SStructNeighborDir(neighbor);
   ilower_mapped = hypre_SStructNeighborILower(neighbor);
   hypre_CopyIndex(index_map, coord);
   hypre_CopyIndex(index_dir, dir);
   for (d = 0; d < ndim; d++)
   {
      dd = coord[d];
      tdir = dir[d];
      /* this effectively sorts nbor_ilower and nbor_iupper */
      if (hypre_IndexD(nbor_ilower, dd) > hypre_IndexD(nbor_iupper, dd))
      {
         tdir = -tdir;
      }
      if (tdir > 0)
      {
         hypre_IndexD(ilower_mapped, dd) = hypre_IndexD(nbor_ilower, dd);
      }
      else
      {
         hypre_IndexD(ilower_mapped, dd) = hypre_IndexD(nbor_iupper, dd);
      }
   }
   for (d = ndim; d < 3; d++)
   {
      hypre_IndexD(coord, d) = d;
      hypre_IndexD(dir, d) = 1;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * DEFUNCT - Use SetNeighborPart instead!
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridSetNeighborBox( HYPRE_SStructGrid  grid,
                                 int                part,
                                 int               *ilower,
                                 int               *iupper,
                                 int                nbor_part,
                                 int               *nbor_ilower,
                                 int               *nbor_iupper,
                                 int               *index_map )
{
   int                      ndim       = hypre_SStructGridNDim(grid);
   int                     *nneighbors = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors  = hypre_SStructGridNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;

   hypre_Box               *box;
   hypre_Index              cilower;
   hypre_Index              ciupper;
   hypre_IndexRef           coord, dir;
   int                      memchunk = 10;
   int                      d, dd;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = hypre_TReAlloc(neighbors[part], hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk));
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nneighbors[part]++;

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxSetExtents(box, cilower, ciupper);

   hypre_SStructNeighborPart(neighbor) = nbor_part;

   hypre_CopyToCleanIndex(nbor_ilower, ndim,
                          hypre_SStructNeighborILower(neighbor));

   coord = hypre_SStructNeighborCoord(neighbor);
   hypre_CopyIndex(index_map, coord);
   for (d = ndim; d < 3; d++)
   {
      hypre_IndexD(coord, d) = d;
   }

   dir = hypre_SStructNeighborDir(neighbor);
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(dir, d) = 1;

      if (d < ndim)
      {
         dd = coord[d];
         if (hypre_IndexD(nbor_ilower, dd) > hypre_IndexD(nbor_iupper, dd))
         {
            hypre_IndexD(dir, d) = -1;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *--------------------------------------------------------------------------*/

#if 0
int
HYPRE_SStructGridAddUnstructuredPart( HYPRE_SStructGrid grid,
                                      int              ilower,
                                      int              iupper )
{
   hypre_SStructGridAssemble(grid);

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridAssemble( HYPRE_SStructGrid grid )
{
   int                      ndim        = hypre_SStructGridNDim(grid);
   int                      nparts      = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids      = hypre_SStructGridPGrids(grid);
   int                     *nneighbors  = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors   = hypre_SStructGridNeighbors(grid);
   int                    **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_SStructNeighbor   *vneighbor;
   hypre_BoxArrayArray     *nbor_boxes;
   hypre_BoxArray          *nbor_boxa;
   hypre_Box               *nbor_box;
   hypre_BoxArray          *sub_boxa;
   hypre_BoxArray          *tmp_boxa;
   hypre_Box               *map_box;
   hypre_Box               *int_box;
   hypre_BoxMapEntry      **entries;
   int                      nentries;
   int                     *ilower, *coord, *dir, *box_imin;
   hypre_Index              imin0, imin1;

   hypre_SStructPGrid      *pgrid;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;
   int                      nvars;
   int                      part, nbor_part, var, nbor_var, i, b, vb, t, d, nd;

   /*-------------------------------------------------------------
    * set pneighbors for each pgrid info to crop pgrids
    *-------------------------------------------------------------*/

   /*
    * ZTODO: Note that if neighbor boxes are not first intersected with
    * the global grid, then local pgrid info may be incorrectly cropped.
    * This would occur if users pass in neighbor extents that do not
    * actually live anywhere on the global grid.
    *
    * This is not an issue for cell-centered variables.
    */ 

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      for (i = 0; i < nneighbors[part]; i++)
      {
         neighbor = &neighbors[part][i];

         /* if this part is not the owner of the shared data */
         if ( part > hypre_SStructNeighborPart(neighbor) )
         {
            hypre_SStructPGridSetPNeighbor(pgrid,
                                           hypre_SStructNeighborBox(neighbor));
         }
      }
   }

   /*-------------------------------------------------------------
    * assemble the pgrids
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPGridAssemble(pgrids[part]);
   }
   
   /*-------------------------------------------------------------
    * re-organize u-variables to reference via local cell rank
    *-------------------------------------------------------------*/

   /* TODO */

   /*-------------------------------------------------------------
    * determine a unique u-variable data distribution
    *-------------------------------------------------------------*/

   /* TODO */

   /*-------------------------------------------------------------
    * set up the size info
    * GEC0902 calculation of the local ghost size for grid
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      hypre_SStructGridLocalSize(grid)   += hypre_SStructPGridLocalSize(pgrid);
      hypre_SStructGridGlobalSize(grid)  += hypre_SStructPGridGlobalSize(pgrid);
      hypre_SStructGridGhlocalSize(grid) += hypre_SStructPGridGhlocalSize(pgrid);
   }

   /*-------------------------------------------------
    * Assemble the map info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleMaps(grid);

   /*-------------------------------------------------
    * Set up neighbor info
    *-------------------------------------------------*/

   nbor_box = hypre_BoxCreate();
   map_box  = hypre_BoxCreate();
   int_box  = hypre_BoxCreate();

   nvneighbors = hypre_TAlloc(int *, nparts);
   vneighbors  = hypre_TAlloc(hypre_SStructNeighbor **, nparts);

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);
      nvneighbors[part] = hypre_TAlloc(int, nvars);
      vneighbors[part]  = hypre_TAlloc(hypre_SStructNeighbor *, nvars);

      nbor_boxes = hypre_BoxArrayArrayCreate(nneighbors[part]);
      for (var = 0; var < nvars; var++)
      {
         t = vartypes[var];
         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffset);

         /* Intersect neighbor boxes with appropriate PGrid */
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor  = &neighbors[part][b];
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            hypre_BoxArraySetSize(nbor_boxa, 0);

            hypre_CopyBox(hypre_SStructNeighborBox(neighbor), nbor_box);
            hypre_CopyIndex(
               hypre_BoxIMin(hypre_SStructNeighborBox(neighbor)), imin0);
            hypre_CopyIndex(hypre_SStructNeighborILower(neighbor), imin1);
            coord = hypre_SStructNeighborCoord(neighbor);
            dir   = hypre_SStructNeighborDir(neighbor);

            /* change variable type; it's important to change imin0 & imin1 */
            for (d = 0; d < 3; d++)
            {
               nd = coord[d];
               hypre_BoxIMinD(nbor_box, d) -= hypre_IndexD(varoffset, d);
               hypre_IndexD(imin0, d) -= hypre_IndexD(varoffset, d);
               if (dir[d] > 0)
               {
                  hypre_IndexD(imin1, nd) -= hypre_IndexD(varoffset, d);
               }
            }

            /* map to neighbor part index space */
            hypre_SStructBoxToNborBox(nbor_box, imin0, imin1, coord, dir);
            hypre_SStructVarToNborVar(grid, part, var, coord, &nbor_var);

            nbor_part = hypre_SStructNeighborPart(neighbor);
            hypre_BoxMapIntersect(hypre_SStructGridMap(grid, nbor_part, nbor_var),
                                  hypre_BoxIMin(nbor_box),
                                  hypre_BoxIMax(nbor_box),
                                  &entries, &nentries );

            for (i = 0; i < nentries; i++)
            {
               hypre_BoxMapEntryGetExtents(entries[i],
                                           hypre_BoxIMin(map_box),
                                           hypre_BoxIMax(map_box));

               hypre_IntersectBoxes(nbor_box, map_box, int_box);
               if (hypre_BoxVolume(int_box))
               {
                  /* map back from neighbor part index space */
                  hypre_SStructNborBoxToBox(int_box, imin0, imin1, coord, dir);

                  hypre_AppendBox(int_box, nbor_boxa);
               }
            }
            hypre_TFree(entries);
         }

         /* Make sure that the neighbor boxes don't overlap */
         tmp_boxa = hypre_BoxArrayCreate(0);
         for (b = 0; b < nneighbors[part]; b++)
         {
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            for (i = 0; i < b; i++)
            {
               sub_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, i);
               hypre_SubtractBoxArrays(nbor_boxa, sub_boxa, tmp_boxa);
            }
         }
         hypre_BoxArrayDestroy(tmp_boxa);

         /* Compute new number of vneighbors, nvneighbors */
         nvneighbors[part][var] = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            nvneighbors[part][var] += hypre_BoxArraySize(nbor_boxa);
         }

         /* Set up vneighbors */
         vneighbors[part][var] =
            hypre_TAlloc(hypre_SStructNeighbor, nvneighbors[part][var]);
         vb = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor  = &neighbors[part][b];
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);

            hypre_CopyIndex(
               hypre_BoxIMin(hypre_SStructNeighborBox(neighbor)), imin0);
            hypre_CopyIndex(hypre_SStructNeighborILower(neighbor), imin1);
            coord = hypre_SStructNeighborCoord(neighbor);
            dir   = hypre_SStructNeighborDir(neighbor);

            /* change variable type as above ('nbor_boxa' already changed) */
            for (d = 0; d < 3; d++)
            {
               nd = coord[d];
               hypre_IndexD(imin0, d) -= hypre_IndexD(varoffset, d);
               if (dir[d] > 0)
               {
                  hypre_IndexD(imin1, nd) -= hypre_IndexD(varoffset, d);
               }
            }

            hypre_ForBoxI(i, nbor_boxa)
               {
                  vneighbor = &vneighbors[part][var][vb];

                  hypre_CopyBox(hypre_BoxArrayBox(nbor_boxa, i),
                                hypre_SStructNeighborBox(vneighbor));
                  hypre_SStructNeighborPart(vneighbor) =
                     hypre_SStructNeighborPart(neighbor);
                  hypre_CopyIndex(imin1, hypre_SStructNeighborILower(vneighbor));
                  hypre_CopyIndex(coord, hypre_SStructNeighborCoord(vneighbor));
                  hypre_CopyIndex(dir, hypre_SStructNeighborDir(vneighbor));

                  /* compute correct vneighbor ilower based on box imin */
                  box_imin = hypre_BoxIMin(hypre_SStructNeighborBox(vneighbor));
                  ilower = hypre_SStructNeighborILower(vneighbor);
                  for (d = 0; d < 3; d++)
                  {
                     nd = coord[d];
                     ilower[nd] = imin1[nd] + (box_imin[d] - imin0[d]) * dir[d];
                  }

                  vb++;
               }
         }
      }

      hypre_BoxArrayArrayDestroy(nbor_boxes);
   }

   hypre_SStructGridNVNeighbors(grid) = nvneighbors;
   hypre_SStructGridVNeighbors(grid)  = vneighbors;

   hypre_BoxDestroy(nbor_box);
   hypre_BoxDestroy(map_box);
   hypre_BoxDestroy(int_box);

   /*-------------------------------------------------
    * Assemble the neighbor map info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleNborMaps(grid);

   /*-------------------------------------------------
    * Compute the CommInfo component of the grid
    *-------------------------------------------------*/

   hypre_SStructGridCreateCommInfo(grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridSetPeriodic( HYPRE_SStructGrid  grid,
                              int                part,
                              int               *periodic )
{
   hypre_SStructPGrid *pgrid          = hypre_SStructGridPGrid(grid, part);
   hypre_IndexRef      pgrid_periodic = hypre_SStructPGridPeriodic(pgrid);
   int                 d;

   for (d = 0; d < hypre_SStructGridNDim(grid); d++)
   {
      hypre_IndexD(pgrid_periodic, d) = periodic[d];
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * GEC0902 a placeholder for a internal function that will set ghosts in each
 * of the sgrids of the grid
 *
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructGridSetNumGhost( HYPRE_SStructGrid grid,
                              int            *num_ghost)       
{
   hypre_SStructGridSetNumGhost(grid, num_ghost);

   return hypre_error_flag;
}

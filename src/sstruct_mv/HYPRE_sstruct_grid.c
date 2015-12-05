/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.24 $
 ***********************************************************************EHEADER*/


/* 9/09 AB - modified all functions to use the box manager */

/******************************************************************************
 *
 * HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridCreate( MPI_Comm           comm,
                         HYPRE_Int          ndim,
                         HYPRE_Int          nparts,
                         HYPRE_SStructGrid *grid_ptr )
{
   hypre_SStructGrid       *grid;
   hypre_SStructPGrid     **pgrids;
   hypre_SStructPGrid      *pgrid;
   HYPRE_Int               *nneighbors;
   hypre_SStructNeighbor  **neighbors;
   hypre_Index            **nbor_offsets;
   HYPRE_Int               *fem_nvars;
   HYPRE_Int              **fem_vars;
   hypre_Index            **fem_offsets;
   HYPRE_Int                i;

   grid = hypre_TAlloc(hypre_SStructGrid, 1);

   hypre_SStructGridComm(grid)   = comm;
   hypre_SStructGridNDim(grid)   = ndim;
   hypre_SStructGridNParts(grid) = nparts;
   pgrids = hypre_TAlloc(hypre_SStructPGrid *, nparts);
   nneighbors    = hypre_TAlloc(HYPRE_Int, nparts);
   neighbors     = hypre_TAlloc(hypre_SStructNeighbor *, nparts);
   nbor_offsets  = hypre_TAlloc(hypre_Index *, nparts);
   fem_nvars     = hypre_TAlloc(HYPRE_Int, nparts);
   fem_vars      = hypre_TAlloc(HYPRE_Int *, nparts);
   fem_offsets   = hypre_TAlloc(hypre_Index *, nparts);
   for (i = 0; i < nparts; i++)
   {
      hypre_SStructPGridCreate(comm, ndim, &pgrid);
      pgrids[i] = pgrid;
      nneighbors[i]    = 0;
      neighbors[i]     = NULL;
      nbor_offsets[i]  = NULL;
      fem_nvars[i]     = 0;
      fem_vars[i]      = NULL;
      fem_offsets[i]   = NULL;
   }
   hypre_SStructGridPGrids(grid)  = pgrids;
   hypre_SStructGridNNeighbors(grid)  = nneighbors;
   hypre_SStructGridNeighbors(grid)   = neighbors;
   hypre_SStructGridNborOffsets(grid) = nbor_offsets;
   hypre_SStructGridNUCVars(grid) = 0;
   hypre_SStructGridUCVars(grid)  = NULL;
   hypre_SStructGridFEMNVars(grid)   = fem_nvars;
   hypre_SStructGridFEMVars(grid)    = fem_vars;
   hypre_SStructGridFEMOffsets(grid) = fem_offsets;

   hypre_SStructGridBoxManagers(grid) = NULL;
   hypre_SStructGridNborBoxManagers(grid) = NULL;

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

HYPRE_Int
HYPRE_SStructGridDestroy( HYPRE_SStructGrid grid )
{
   HYPRE_Int                      nparts;
   hypre_SStructPGrid           **pgrids;
   HYPRE_Int                     *nneighbors;
   hypre_SStructNeighbor        **neighbors;
   hypre_Index                  **nbor_offsets;
   HYPRE_Int                    **nvneighbors;
   hypre_SStructNeighbor       ***vneighbors;
   hypre_SStructCommInfo        **vnbor_comm_info;
   HYPRE_Int                      vnbor_ncomms;
   HYPRE_Int                     *fem_nvars;
   HYPRE_Int                    **fem_vars;
   hypre_Index                  **fem_offsets;
   hypre_BoxManager            ***managers;
   hypre_BoxManager            ***nbor_managers;
   HYPRE_Int                      nvars;
   HYPRE_Int                      part, var, i;

   if (grid)
   {
      hypre_SStructGridRefCount(grid) --;
      if (hypre_SStructGridRefCount(grid) == 0)
      {
         nparts  = hypre_SStructGridNParts(grid);
         pgrids  = hypre_SStructGridPGrids(grid);
         nneighbors   = hypre_SStructGridNNeighbors(grid);
         neighbors    = hypre_SStructGridNeighbors(grid);
         nbor_offsets = hypre_SStructGridNborOffsets(grid);
         nvneighbors = hypre_SStructGridNVNeighbors(grid);
         vneighbors  = hypre_SStructGridVNeighbors(grid);
         vnbor_comm_info = hypre_SStructGridVNborCommInfo(grid);
         vnbor_ncomms = hypre_SStructGridVNborNComms(grid);
         fem_nvars   = hypre_SStructGridFEMNVars(grid);
         fem_vars    = hypre_SStructGridFEMVars(grid);
         fem_offsets = hypre_SStructGridFEMOffsets(grid);
         managers  = hypre_SStructGridBoxManagers(grid);
         nbor_managers  = hypre_SStructGridNborBoxManagers(grid);

         for (part = 0; part < nparts; part++)
         {
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               hypre_TFree(vneighbors[part][var]);
               hypre_BoxManDestroy(managers[part][var]);
               hypre_BoxManDestroy(nbor_managers[part][var]);
            }
            hypre_TFree(neighbors[part]);
            hypre_TFree(nbor_offsets[part]);
            hypre_TFree(nvneighbors[part]);
            hypre_TFree(vneighbors[part]);
            hypre_SStructPGridDestroy(pgrids[part]);
            hypre_TFree(fem_vars[part]);
            hypre_TFree(fem_offsets[part]);
            hypre_TFree(managers[part]);
            hypre_TFree(nbor_managers[part]);
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
         hypre_TFree(nbor_offsets);
         hypre_TFree(fem_nvars);
         hypre_TFree(fem_vars);
         hypre_TFree(fem_offsets);
         hypre_TFree(nvneighbors);
         hypre_TFree(vneighbors);
         hypre_TFree(vnbor_comm_info);
         hypre_TFree(managers);
         hypre_TFree(nbor_managers);
         hypre_TFree(grid);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetExtents( HYPRE_SStructGrid  grid,
                             HYPRE_Int          part,
                             HYPRE_Int         *ilower,
                             HYPRE_Int         *iupper )
{
   HYPRE_Int            ndim  = hypre_SStructGridNDim(grid);
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

HYPRE_Int HYPRE_SStructGridSetVariables( HYPRE_SStructGrid      grid,
                                   HYPRE_Int              part,
                                   HYPRE_Int              nvars,
                                   HYPRE_SStructVariable *vartypes )
{
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);

   hypre_SStructPGridSetVariables(pgrid, nvars, vartypes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridAddVariables( HYPRE_SStructGrid      grid,
                               HYPRE_Int              part,
                               HYPRE_Int             *index,
                               HYPRE_Int              nvars,
                               HYPRE_SStructVariable *vartypes )
{
   HYPRE_Int            ndim    = hypre_SStructGridNDim(grid);
   HYPRE_Int            nucvars = hypre_SStructGridNUCVars(grid);
   hypre_SStructUCVar **ucvars  = hypre_SStructGridUCVars(grid);
   hypre_SStructUCVar  *ucvar;

   HYPRE_Int            memchunk = 1000;
   HYPRE_Int            i;

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
 * If ordering == NULL, use a default ordering.  This feature is mainly for
 * internal implementation reasons.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetFEMOrdering( HYPRE_SStructGrid  grid,
                                 HYPRE_Int          part,
                                 HYPRE_Int         *ordering )
{
   HYPRE_Int               ndim     = hypre_SStructGridNDim(grid);
   hypre_SStructPGrid     *pgrid    = hypre_SStructGridPGrid(grid, part);
   HYPRE_Int               nvars    = hypre_SStructPGridNVars(pgrid);
   HYPRE_SStructVariable  *vartypes = hypre_SStructPGridVarTypes(pgrid);
   HYPRE_Int               fem_nvars;
   HYPRE_Int              *fem_vars;
   hypre_Index            *fem_offsets;
   hypre_Index             varoffset;
   HYPRE_Int               i, j, d, nv, *block, off[3], loop[3];
   HYPRE_Int               clean = 0;

   /* compute fem_nvars */
   fem_nvars = 0;
   for (i = 0; i < nvars; i++)
   {
      nv = 1;
      hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
      for (d = 0; d < ndim; d++)
      {
         if (varoffset[d])
         {
            nv *= 2;
         }
      }
      fem_nvars += nv;
   }

   /* set default ordering */
   if (ordering == NULL)
   {
      clean = 1;
      ordering = hypre_TAlloc(HYPRE_Int, (1+ndim)*fem_nvars);
      j = 0;
      for (i = 0; i < nvars; i++)
      {
         hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            loop[d] = 0;
            if ((varoffset[d] != 0) && (d < ndim))
            {
               loop[d] = 1;
            }
         }
         for (off[2] = -loop[2]; off[2] <= loop[2]; off[2]+=2)
         {
            for (off[1] = -loop[1]; off[1] <= loop[1]; off[1]+=2)
            {
               for (off[0] = -loop[0]; off[0] <= loop[0]; off[0]+=2)
               {
                  block = &ordering[(1+ndim)*j];
                  block[0] = i;
                  for (d = 0; d < ndim; d++)
                  {
                     block[1+d] = off[d];
                  }
                  j++;
               }
            }
         }
      }
   }

   fem_vars    = hypre_TAlloc(HYPRE_Int, fem_nvars);
   fem_offsets = hypre_TAlloc(hypre_Index, fem_nvars);
   for (i = 0; i < fem_nvars; i++)
   {
      block = &ordering[(1+ndim)*i];
      fem_vars[i] = block[0];
      hypre_ClearIndex(fem_offsets[i]);
      for (d = 0; d < ndim; d++)
      {
         /* modify the user offsets to contain only 0's and -1's */
         if (block[1+d] < 0)
         {
            hypre_IndexD(fem_offsets[i], d) = -1;
         }
      }
   }

   hypre_SStructGridFEMPNVars(grid, part)   = fem_nvars;
   hypre_SStructGridFEMPVars(grid, part)    = fem_vars;
   hypre_SStructGridFEMPOffsets(grid, part) = fem_offsets;

   if (clean)
   {
      hypre_TFree(ordering);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetNeighborPart( HYPRE_SStructGrid  grid,
                                  HYPRE_Int          part,
                                  HYPRE_Int         *ilower,
                                  HYPRE_Int         *iupper,
                                  HYPRE_Int          nbor_part,
                                  HYPRE_Int         *nbor_ilower,
                                  HYPRE_Int         *nbor_iupper,
                                  HYPRE_Int         *index_map,
                                  HYPRE_Int         *index_dir )
{
   HYPRE_Int                ndim         = hypre_SStructGridNDim(grid);
   HYPRE_Int               *nneighbors   = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors    = hypre_SStructGridNeighbors(grid);
   hypre_Index            **nbor_offsets = hypre_SStructGridNborOffsets(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_IndexRef           nbor_offset;

   hypre_Box               *box;
   hypre_Index              cilower;
   hypre_Index              ciupper;
   hypre_IndexRef           coord, dir, ilower_mapped;
   HYPRE_Int                memchunk = 10;
   HYPRE_Int                d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = hypre_TReAlloc(neighbors[part], hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk));
      nbor_offsets[part] = hypre_TReAlloc(nbor_offsets[part], hypre_Index,
                                          (nneighbors[part] + memchunk));
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];
   nneighbors[part]++;

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxSetExtents(box, cilower, ciupper);
   hypre_ClearIndex(nbor_offset);

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
      hypre_IndexD(ilower_mapped, d) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetSharedPart( HYPRE_SStructGrid  grid,
                                HYPRE_Int          part,
                                HYPRE_Int         *ilower,
                                HYPRE_Int         *iupper,
                                HYPRE_Int         *offset,
                                HYPRE_Int          shared_part,
                                HYPRE_Int         *shared_ilower,
                                HYPRE_Int         *shared_iupper,
                                HYPRE_Int         *shared_offset,
                                HYPRE_Int         *index_map,
                                HYPRE_Int         *index_dir )
{
   HYPRE_Int                ndim       = hypre_SStructGridNDim(grid);
   HYPRE_Int               *nneighbors = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors  = hypre_SStructGridNeighbors(grid);
   hypre_Index            **nbor_offsets = hypre_SStructGridNborOffsets(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_IndexRef           nbor_offset;

   hypre_Box               *box;
   hypre_Index              cilower;
   hypre_Index              ciupper;
   hypre_IndexRef           coord, dir, ilower_mapped;
   HYPRE_Int                offset_mapped[3];
   HYPRE_Int                memchunk = 10;
   HYPRE_Int                d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = hypre_TReAlloc(neighbors[part], hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk));
      nbor_offsets[part] = hypre_TReAlloc(nbor_offsets[part], hypre_Index,
                                          (nneighbors[part] + memchunk));
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];
   nneighbors[part]++;

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxSetExtents(box, cilower, ciupper);
   hypre_CopyToCleanIndex(offset, ndim, nbor_offset);

   hypre_SStructNeighborPart(neighbor) = shared_part;

   coord = hypre_SStructNeighborCoord(neighbor);
   dir = hypre_SStructNeighborDir(neighbor);
   ilower_mapped = hypre_SStructNeighborILower(neighbor);
   hypre_CopyIndex(index_map, coord);
   hypre_CopyIndex(index_dir, dir);
   for (d = 0; d < ndim; d++)
   {
      dd = coord[d];
      tdir = dir[d];
      /* this effectively sorts shared_ilower and shared_iupper */
      if (hypre_IndexD(shared_ilower, dd) > hypre_IndexD(shared_iupper, dd))
      {
         tdir = -tdir;
      }
      if (tdir > 0)
      {
         hypre_IndexD(ilower_mapped, dd) = hypre_IndexD(shared_ilower, dd);
      }
      else
      {
         hypre_IndexD(ilower_mapped, dd) = hypre_IndexD(shared_iupper, dd);
      }
      /* map the offset to the neighbor part and adjust ilower_mapped */
      offset_mapped[dd] = offset[d]*dir[d];
      if (offset_mapped[dd] != shared_offset[dd])
      {
         hypre_IndexD(ilower_mapped, dd) -= offset_mapped[dd];
      }
   }
   for (d = ndim; d < 3; d++)
   {
      hypre_IndexD(coord, d) = d;
      hypre_IndexD(dir, d) = 1;
      hypre_IndexD(ilower_mapped, d) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *--------------------------------------------------------------------------*/

#if 0
HYPRE_Int
HYPRE_SStructGridAddUnstructuredPart( HYPRE_SStructGrid grid,
                                      HYPRE_Int        ilower,
                                      HYPRE_Int        iupper )
{
   hypre_SStructGridAssemble(grid);

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridAssemble( HYPRE_SStructGrid grid )
{
   HYPRE_Int                ndim         = hypre_SStructGridNDim(grid);
   HYPRE_Int                nparts       = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids       = hypre_SStructGridPGrids(grid);
   HYPRE_Int               *nneighbors   = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors    = hypre_SStructGridNeighbors(grid);
   hypre_Index            **nbor_offsets = hypre_SStructGridNborOffsets(grid);
   HYPRE_Int              **nvneighbors  = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor ***vneighbors   = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_IndexRef           nbor_offset;
   hypre_SStructNeighbor   *vneighbor;
   hypre_Box               *box;
   HYPRE_Int               *ilower, *coord, *dir;

   hypre_SStructPGrid      *pgrid;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;
   HYPRE_Int                nvars;
   HYPRE_Int                part, var, b, vb, d, i, valid;

   /*-------------------------------------------------------------
    * if I own no data on some part, prune that part's neighbor info
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      if (hypre_StructGridNumBoxes(hypre_SStructPGridCellSGrid(pgrid)) == 0)
      {
         nneighbors[part] = 0;
         hypre_TFree(neighbors[part]);
         hypre_TFree(nbor_offsets[part]);
      }
   }

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
      for (b = 0; b < nneighbors[part]; b++)
      {
         neighbor = &neighbors[part][b];
         nbor_offset = nbor_offsets[part][b];

         /* if this part is not the owner of the shared data */
         if ( part > hypre_SStructNeighborPart(neighbor) )
         {
            hypre_SStructPGridSetPNeighbor(
               pgrid, hypre_SStructNeighborBox(neighbor), nbor_offset);
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
    * Set up the FEM ordering information
    *-------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      if (hypre_SStructGridFEMPNVars(grid, part) == 0)
      {
         /* use the default ordering */
         HYPRE_SStructGridSetFEMOrdering(grid, part, NULL);
      }
   }

   /*-------------------------------------------------
    * Set up vneighbor info
    *-------------------------------------------------*/

   nvneighbors = hypre_TAlloc(HYPRE_Int *, nparts);
   vneighbors  = hypre_TAlloc(hypre_SStructNeighbor **, nparts);

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);
      nvneighbors[part] = hypre_TAlloc(HYPRE_Int, nvars);
      vneighbors[part]  = hypre_TAlloc(hypre_SStructNeighbor *, nvars);

      for (var = 0; var < nvars; var++)
      {
         vneighbors[part][var] = hypre_TAlloc(hypre_SStructNeighbor,
                                              nneighbors[part]);

         hypre_SStructVariableGetOffset((hypre_SStructVariable) vartypes[var],
                                        ndim, varoffset);

         vb = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor    = &neighbors[part][b];
            nbor_offset = nbor_offsets[part][b];
            vneighbor   = &vneighbors[part][var][vb];

            /* set pointers to vneighbor data */
            box    = hypre_SStructNeighborBox(vneighbor);
            ilower = hypre_SStructNeighborILower(vneighbor);
            coord  = hypre_SStructNeighborCoord(vneighbor);
            dir    = hypre_SStructNeighborDir(vneighbor);
            /* copy neighbor data into vneighbor */
            hypre_CopyBox(hypre_SStructNeighborBox(neighbor), box);
            hypre_SStructNeighborPart(vneighbor) =
               hypre_SStructNeighborPart(neighbor);
            hypre_CopyIndex(hypre_SStructNeighborILower(neighbor), ilower);
            hypre_CopyIndex(hypre_SStructNeighborCoord(neighbor), coord);
            hypre_CopyIndex(hypre_SStructNeighborDir(neighbor), dir);
            hypre_SStructCellBoxToVarBox(box, nbor_offset, varoffset, &valid);
            /* it's important to change ilower */
            for (d = 0; d < 3; d++)
            {
               /* Compare the imin of the neighbor cell box ('i') to its imin
                * value after being converted to a variable box ('IMin(box,d)').
                * If the coordinates in the two parts move in the same direction
                * (i.e., dir[d] > 0) and the local imin changed, then also
                * change the corresponding neighbor ilower.  If the coordinates
                * in the two parts move in opposite directions and the local
                * imin did not change, then change the corresponding neighbor
                * ilower based on the value of 'varoffset'. */
               i = hypre_BoxIMinD(hypre_SStructNeighborBox(neighbor), d);
               if (((dir[d] > 0) && (hypre_BoxIMinD(box, d) != i)) ||
                   ((dir[d] < 0) && (hypre_BoxIMinD(box, d) == i)))
               {
                  hypre_IndexD(ilower, coord[d]) -= hypre_IndexD(varoffset, d);
               }
            }
            /* some variable types may lead to empty variable boxes? */
            if (valid && hypre_BoxVolume(box))
            {
               vb++;
            }

         } /* end of neighbor loop */

         nvneighbors[part][var] = vb;

      } /* end of variables loop */
   } /* end of part loop */

   hypre_SStructGridNVNeighbors(grid) = nvneighbors;
   hypre_SStructGridVNeighbors(grid)  = vneighbors;

   /*-------------------------------------------------
    * Assemble the box manager info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleBoxManagers(grid);

   /*-------------------------------------------------
    * Assemble the neighbor box manager info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleNborBoxManagers(grid);

   /*-------------------------------------------------
    * Compute the CommInfo component of the grid
    *-------------------------------------------------*/

   hypre_SStructGridCreateCommInfo(grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetPeriodic( HYPRE_SStructGrid  grid,
                              HYPRE_Int          part,
                              HYPRE_Int         *periodic )
{
   hypre_SStructPGrid *pgrid          = hypre_SStructGridPGrid(grid, part);
   hypre_IndexRef      pgrid_periodic = hypre_SStructPGridPeriodic(pgrid);
   HYPRE_Int           d;

   for (d = 0; d < hypre_SStructGridNDim(grid); d++)
   {
      hypre_IndexD(pgrid_periodic, d) = periodic[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC0902 a placeholder for a internal function that will set ghosts in each
 * of the sgrids of the grid
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridSetNumGhost( HYPRE_SStructGrid grid,
                              HYPRE_Int      *num_ghost)       
{
   hypre_SStructGridSetNumGhost(grid, num_ghost);

   return hypre_error_flag;
}






/*--------------------------------------------------------------------------
 * Like HYPRE_SStructGridSetVariables, but do one variable at a time.
 * Nevertheless, you still must provide nvars, for memory allocation.
 *
 * RDF: Why is this routine here?  It's not in the header file?
 *      Looks like it's purely babel related.
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_SStructGridSetVariable( HYPRE_SStructGrid      grid,
                                  HYPRE_Int              part,
                                  HYPRE_Int              var,
                                  HYPRE_Int              nvars,
                                  HYPRE_SStructVariable  vartype )
{
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);

   hypre_SStructPGridSetVariable( pgrid, var, nvars, vartype );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Like HYPRE_SStructGridAddVariables, but just one variable at a time.
 *
 * RDF: Why is this routine here?  It's not in the header file?
 *      Looks like it's purely babel related.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGridAddVariable( HYPRE_SStructGrid      grid,
                              HYPRE_Int              part,
                              HYPRE_Int             *index,
                              HYPRE_Int              var,
                              HYPRE_SStructVariable  vartype )
{
   HYPRE_Int            ndim    = hypre_SStructGridNDim(grid);
   HYPRE_Int            nucvars = hypre_SStructGridNUCVars(grid);
   hypre_SStructUCVar **ucvars  = hypre_SStructGridUCVars(grid);
   hypre_SStructUCVar  *ucvar;

   HYPRE_Int            memchunk = 1000;
   HYPRE_Int            nvars = 1;  /* each ucvar gets only one variable */

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

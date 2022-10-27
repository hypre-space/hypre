/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* 9/09 AB - modified all functions to use the box manager */

/******************************************************************************
 *
 * HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

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
   HYPRE_Int                num_ghost[2 * HYPRE_MAXDIM];
   HYPRE_Int                i;

   grid = hypre_TAlloc(hypre_SStructGrid,  1, HYPRE_MEMORY_HOST);

   hypre_SStructGridComm(grid)   = comm;
   hypre_SStructGridNDim(grid)   = ndim;
   hypre_SStructGridNParts(grid) = nparts;
   pgrids = hypre_TAlloc(hypre_SStructPGrid *,  nparts, HYPRE_MEMORY_HOST);
   nneighbors    = hypre_TAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
   neighbors     = hypre_TAlloc(hypre_SStructNeighbor *,  nparts, HYPRE_MEMORY_HOST);
   nbor_offsets  = hypre_TAlloc(hypre_Index *,  nparts, HYPRE_MEMORY_HOST);
   fem_nvars     = hypre_TAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
   fem_vars      = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   fem_offsets   = hypre_TAlloc(hypre_Index *,  nparts, HYPRE_MEMORY_HOST);
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

   /* Initialize num ghost */
   for (i = 0; i < 2 * ndim; i++)
   {
      num_ghost[i] = 1;
   }
   hypre_SStructGridSetNumGhost(grid, num_ghost);

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
               hypre_TFree(vneighbors[part][var], HYPRE_MEMORY_HOST);
               hypre_BoxManDestroy(managers[part][var]);
               hypre_BoxManDestroy(nbor_managers[part][var]);
            }
            hypre_TFree(neighbors[part], HYPRE_MEMORY_HOST);
            hypre_TFree(nbor_offsets[part], HYPRE_MEMORY_HOST);
            hypre_TFree(nvneighbors[part], HYPRE_MEMORY_HOST);
            hypre_TFree(vneighbors[part], HYPRE_MEMORY_HOST);
            hypre_SStructPGridDestroy(pgrids[part]);
            hypre_TFree(fem_vars[part], HYPRE_MEMORY_HOST);
            hypre_TFree(fem_offsets[part], HYPRE_MEMORY_HOST);
            hypre_TFree(managers[part], HYPRE_MEMORY_HOST);
            hypre_TFree(nbor_managers[part], HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < vnbor_ncomms; i++)
         {
            hypre_CommInfoDestroy(
               hypre_SStructCommInfoCommInfo(vnbor_comm_info[i]));
            hypre_TFree(vnbor_comm_info[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(vnbor_comm_info, HYPRE_MEMORY_HOST);
         hypre_TFree(pgrids, HYPRE_MEMORY_HOST);
         hypre_TFree(nneighbors, HYPRE_MEMORY_HOST);
         hypre_TFree(neighbors, HYPRE_MEMORY_HOST);
         hypre_TFree(nbor_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_nvars, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_vars, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(nvneighbors, HYPRE_MEMORY_HOST);
         hypre_TFree(vneighbors, HYPRE_MEMORY_HOST);
         hypre_TFree(vnbor_comm_info, HYPRE_MEMORY_HOST);
         hypre_TFree(managers, HYPRE_MEMORY_HOST);
         hypre_TFree(nbor_managers, HYPRE_MEMORY_HOST);
         hypre_TFree(grid, HYPRE_MEMORY_HOST);
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

HYPRE_Int
HYPRE_SStructGridSetVariables( HYPRE_SStructGrid      grid,
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
      ucvars = hypre_TReAlloc(ucvars,  hypre_SStructUCVar *,
                              (nucvars + memchunk), HYPRE_MEMORY_HOST);
   }

   ucvar = hypre_TAlloc(hypre_SStructUCVar,  1, HYPRE_MEMORY_HOST);
   hypre_SStructUCVarUVars(ucvar) = hypre_TAlloc(hypre_SStructUVar,  nvars, HYPRE_MEMORY_HOST);
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

/* ONLY3D */

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
      ordering = hypre_TAlloc(HYPRE_Int,  (1 + ndim) * fem_nvars, HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 0; i < nvars; i++)
      {
         hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            loop[d] = 0;
            if ((d < ndim) && (varoffset[d] != 0))
            {
               loop[d] = 1;
            }
         }
         for (off[2] = -loop[2]; off[2] <= loop[2]; off[2] += 2)
         {
            for (off[1] = -loop[1]; off[1] <= loop[1]; off[1] += 2)
            {
               for (off[0] = -loop[0]; off[0] <= loop[0]; off[0] += 2)
               {
                  block = &ordering[(1 + ndim) * j];
                  block[0] = i;
                  for (d = 0; d < ndim; d++)
                  {
                     block[1 + d] = off[d];
                  }
                  j++;
               }
            }
         }
      }
   }

   fem_vars    = hypre_TReAlloc(hypre_SStructGridFEMPVars(grid, part), HYPRE_Int, fem_nvars,
                                HYPRE_MEMORY_HOST);
   fem_offsets = hypre_TReAlloc(hypre_SStructGridFEMPOffsets(grid, part), hypre_Index, fem_nvars,
                                HYPRE_MEMORY_HOST);

   for (i = 0; i < fem_nvars; i++)
   {
      block = &ordering[(1 + ndim) * i];
      fem_vars[i] = block[0];
      hypre_SetIndex(fem_offsets[i], 0);
      for (d = 0; d < ndim; d++)
      {
         /* modify the user offsets to contain only 0's and -1's */
         if (block[1 + d] < 0)
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
      hypre_TFree(ordering, HYPRE_MEMORY_HOST);
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
      neighbors[part] = hypre_TReAlloc(neighbors[part],  hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk), HYPRE_MEMORY_HOST);
      nbor_offsets[part] = hypre_TReAlloc(nbor_offsets[part],  hypre_Index,
                                          (nneighbors[part] + memchunk), HYPRE_MEMORY_HOST);
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxInit(box, ndim);
   hypre_BoxSetExtents(box, cilower, ciupper);
   hypre_SetIndex(nbor_offset, 0);

   /* If the neighbor box is empty, return */
   if ( !(hypre_BoxVolume(box) > 0) )
   {
      return hypre_error_flag;
   }

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
   for (d = ndim; d < ndim; d++)
   {
      hypre_IndexD(coord, d) = d;
      hypre_IndexD(dir, d) = 1;
      hypre_IndexD(ilower_mapped, d) = 0;
   }

   nneighbors[part]++;

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
   HYPRE_Int                offset_mapped[HYPRE_MAXDIM];
   HYPRE_Int                memchunk = 10;
   HYPRE_Int                d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = hypre_TReAlloc(neighbors[part],  hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk), HYPRE_MEMORY_HOST);
      nbor_offsets[part] = hypre_TReAlloc(nbor_offsets[part],  hypre_Index,
                                          (nneighbors[part] + memchunk), HYPRE_MEMORY_HOST);
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];

   box = hypre_SStructNeighborBox(neighbor);
   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxInit(box, ndim);
   hypre_BoxSetExtents(box, cilower, ciupper);
   hypre_CopyToCleanIndex(offset, ndim, nbor_offset);

   /* If the neighbor box is empty, return */
   if ( !(hypre_BoxVolume(box) > 0) )
   {
      return hypre_error_flag;
   }

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
      /* Map the offset to the neighbor part and adjust ilower_mapped so that
       * NeighborILower is a direct mapping of NeighborBoxIMin.  This allows us
       * to eliminate shared_offset. */
      offset_mapped[dd] = offset[d] * dir[d];
      if (offset_mapped[dd] != shared_offset[dd])
      {
         hypre_IndexD(ilower_mapped, dd) -= offset_mapped[dd];
      }
   }
   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(coord, d) = d;
      hypre_IndexD(dir, d) = 1;
      hypre_IndexD(ilower_mapped, d) = 0;
   }

   nneighbors[part]++;

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
   HYPRE_Int               *coord, *dir;
   hypre_Index             *fr_roots, *to_roots;
   hypre_BoxArrayArray     *nbor_boxes;
   hypre_BoxArray          *nbor_boxa;
   hypre_BoxArray          *sub_boxa;
   hypre_BoxArray          *tmp_boxa;
   hypre_Box               *nbor_box, *box;
   hypre_SStructPGrid      *pgrid;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;
   HYPRE_Int                nvars;
   HYPRE_Int                part, var, b, vb, d, i, valid;
   HYPRE_Int                nbor_part, sub_part;

   /*-------------------------------------------------------------
    * if I own no data on some part, prune that part's neighbor info
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      if (hypre_StructGridNumBoxes(hypre_SStructPGridCellSGrid(pgrid)) == 0)
      {
         nneighbors[part] = 0;
         hypre_TFree(neighbors[part], HYPRE_MEMORY_HOST);
         hypre_TFree(nbor_offsets[part], HYPRE_MEMORY_HOST);
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

   box = hypre_BoxCreate(ndim);
   tmp_boxa = hypre_BoxArrayCreate(0, ndim);

   nvneighbors = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   vneighbors  = hypre_TAlloc(hypre_SStructNeighbor **,  nparts, HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);
      nvneighbors[part] = hypre_TAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);
      vneighbors[part]  = hypre_TAlloc(hypre_SStructNeighbor *,  nvars, HYPRE_MEMORY_HOST);

      for (var = 0; var < nvars; var++)
      {
         /* Put each new vneighbor box into a BoxArrayArray so we can remove overlap */
         nbor_boxes = hypre_BoxArrayArrayCreate(nneighbors[part], ndim);
         fr_roots = hypre_TAlloc(hypre_Index,  nneighbors[part], HYPRE_MEMORY_HOST);
         to_roots = hypre_TAlloc(hypre_Index,  nneighbors[part], HYPRE_MEMORY_HOST);
         hypre_SStructVariableGetOffset((hypre_SStructVariable) vartypes[var], ndim, varoffset);
         nvneighbors[part][var] = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor    = &neighbors[part][b];
            nbor_offset = nbor_offsets[part][b];

            /* Create var-centered vneighbor box from cell-centered neighbor box */
            hypre_CopyBox(hypre_SStructNeighborBox(neighbor), box);
            hypre_SStructCellBoxToVarBox(box, nbor_offset, varoffset, &valid);
            /* Sometimes we can't construct vneighbor boxes (valid = false).
             * For example, if only faces are shared (see SetSharedPart), then
             * there should be no vneighbor boxes for cell variables.  Note that
             * we ensure nonempty neighbor boxes when they are set up. */
            if (!valid)
            {
               continue;
            }

            /* Save root mapping information for later */
            hypre_CopyIndex(hypre_BoxIMin(box), fr_roots[b]);
            hypre_CopyIndex(hypre_SStructNeighborILower(neighbor), to_roots[b]);

            /* It's important to adjust to_root (ilower) */
            coord = hypre_SStructNeighborCoord(neighbor);
            dir   = hypre_SStructNeighborDir(neighbor);
            for (d = 0; d < ndim; d++)
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
                  hypre_IndexD(to_roots[b], coord[d]) -= hypre_IndexD(varoffset, d);
               }
            }

            /* Add box to the nbor_boxes */
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            hypre_AppendBox(box, nbor_boxa);

            /* Make sure that the nbor_boxes don't overlap */
            nbor_part = hypre_SStructNeighborPart(neighbor);
            for (i = 0; i < b; i++)
            {
               neighbor = &neighbors[part][i];
               sub_part = hypre_SStructNeighborPart(neighbor);
               /* Only subtract boxes on the same neighbor part */
               if (nbor_part == sub_part)
               {
                  sub_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, i);
                  /* nbor_boxa -= sub_boxa */
                  hypre_SubtractBoxArrays(nbor_boxa, sub_boxa, tmp_boxa);
               }
            }

            nvneighbors[part][var] += hypre_BoxArraySize(nbor_boxa);
         }

         /* Set up vneighbors for this (part, var) */
         vneighbors[part][var] = hypre_TAlloc(hypre_SStructNeighbor,  nvneighbors[part][var],
                                              HYPRE_MEMORY_HOST);
         vb = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor  = &neighbors[part][b];
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            nbor_part = hypre_SStructNeighborPart(neighbor);
            coord     = hypre_SStructNeighborCoord(neighbor);
            dir       = hypre_SStructNeighborDir(neighbor);
            hypre_ForBoxI(i, nbor_boxa)
            {
               vneighbor = &vneighbors[part][var][vb];
               nbor_box = hypre_BoxArrayBox(nbor_boxa, i);

               hypre_CopyBox(nbor_box, hypre_SStructNeighborBox(vneighbor));
               hypre_SStructNeighborPart(vneighbor) = nbor_part;
               hypre_SStructIndexToNborIndex(hypre_BoxIMin(nbor_box),
                                             fr_roots[b], to_roots[b], coord, dir, ndim,
                                             hypre_SStructNeighborILower(vneighbor));
               hypre_CopyIndex(coord, hypre_SStructNeighborCoord(vneighbor));
               hypre_CopyIndex(dir, hypre_SStructNeighborDir(vneighbor));

               vb++;
            }

         } /* end of vneighbor box loop */

         hypre_BoxArrayArrayDestroy(nbor_boxes);
         hypre_TFree(fr_roots, HYPRE_MEMORY_HOST);
         hypre_TFree(to_roots, HYPRE_MEMORY_HOST);

      } /* end of variables loop */
   } /* end of part loop */

   hypre_SStructGridNVNeighbors(grid) = nvneighbors;
   hypre_SStructGridVNeighbors(grid)  = vneighbors;

   hypre_BoxArrayDestroy(tmp_boxa);
   hypre_BoxDestroy(box);

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

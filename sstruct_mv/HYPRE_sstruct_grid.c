/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
   int  ierr = 0;

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

   *grid_ptr = grid;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridDestroy( HYPRE_SStructGrid grid )
{
   int ierr = 0;

   int                        nparts;
   hypre_SStructPGrid       **pgrids;
   int                       *nneighbors;
   hypre_SStructNeighbor    **neighbors;
   int                      **nvneighbors;
   hypre_SStructNeighbor   ***vneighbors;
   hypre_BoxMap            ***maps;
   hypre_SStructMapInfo    ***info;
   hypre_SStructNMapInfo   ***ninfo;
   int                        nvars;
   int                        part, var;

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
         hypre_TFree(pgrids);
         hypre_TFree(nneighbors);
         hypre_TFree(neighbors);
         hypre_TFree(nvneighbors);
         hypre_TFree(vneighbors);
         hypre_TFree(maps);
         hypre_TFree(info);
         hypre_TFree(ninfo);
         hypre_TFree(grid);
      }
   }

   return ierr;
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

   return ( hypre_SStructPGridSetExtents(pgrid, cilower, ciupper) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int HYPRE_SStructGridSetVariables( HYPRE_SStructGrid      grid,
                                   int                    part,
                                   int                    nvars,
                                   HYPRE_SStructVariable *vartypes )
{
   hypre_SStructPGrid  *pgrid = hypre_SStructGridPGrid(grid, part);

   return ( hypre_SStructPGridSetVariables(pgrid, nvars, vartypes) );
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
   int  ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
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
   int ierr = 0;

   int                      ndim       = hypre_SStructGridNDim(grid);
   int                     *nneighbors = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors  = hypre_SStructGridNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;

   hypre_Box               *box;
   hypre_Index              cilower;
   hypre_Index              ciupper;
   int                      memchunk = 10;
   int                      d;

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

   hypre_CopyToCleanIndex(index_map, ndim,
                          hypre_SStructNeighborCoord(neighbor));

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(hypre_SStructNeighborDir(neighbor), d) = 1;

      if (d < ndim)
      {
         if (hypre_IndexD(nbor_ilower, d) > hypre_IndexD(nbor_iupper, d))
         {
            hypre_IndexD(hypre_SStructNeighborDir(neighbor), d) = -1;
         }
      }
   }

   return ierr;
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
   return ( hypre_SStructGridAssemble(grid) );
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridAssemble( HYPRE_SStructGrid grid )
{
   int                   ierr = 0;

   int                      ndim        = hypre_SStructGridNDim(grid);
   int                      nparts      = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids      = hypre_SStructGridPGrids(grid);
   int                     *nneighbors  = hypre_SStructGridNNeighbors(grid);
   hypre_SStructNeighbor  **neighbors   = hypre_SStructGridNeighbors(grid);
   int                    **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_SStructNeighbor   *vneighbor;
   hypre_Box               *nbor_box;

   hypre_SStructPGrid      *pgrid;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;
   int                      nvars;
   int                      part, var, i, b, t;

   /*-------------------------------------------------------------
    * set pneighbors for each pgrid info to crop pgrids
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      for (i = 0; i < nneighbors[part]; i++)
      {
         neighbor = &neighbors[part][i];

         if (hypre_SStructNeighborPart(neighbor) < part)
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
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      hypre_SStructGridLocalSize(grid)  += hypre_SStructPGridLocalSize(pgrid);
      hypre_SStructGridGlobalSize(grid) += hypre_SStructPGridGlobalSize(pgrid);
   }

   /*-------------------------------------------------
    * Set up neighbor info
    *
    * ZTODO: This only works for cell-centered variables
    * right now.  To generalize, we need to subtract from
    * each neighbor box the local boxes.  But, then we need
    * to be able to find the offset info, etc.  This may
    * be straightforward.  The ability to use this info to
    * construct communication packages in the VectorGather
    * should also be considered when rewriting.
    *-------------------------------------------------*/

   nvneighbors = hypre_TAlloc(int *, nparts);
   vneighbors  = hypre_TAlloc(hypre_SStructNeighbor **, nparts);

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);
      nvneighbors[part] = hypre_TAlloc(int, nvars);
      vneighbors[part]  = hypre_TAlloc(hypre_SStructNeighbor *, nvars);

      for (var = 0; var < nvars; var++)
      {
         t = vartypes[var];
         vneighbors[part][var] = hypre_TAlloc(hypre_SStructNeighbor,
                                              nneighbors[part]);

         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffset);

         nvneighbors[part][var] = nneighbors[part];
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor  = &neighbors[part][b];
            vneighbor = &vneighbors[part][var][b];

            nbor_box = hypre_SStructNeighborBox(vneighbor);
            hypre_CopyBox(hypre_SStructNeighborBox(neighbor), nbor_box);
            hypre_BoxIMinX(nbor_box) -= hypre_IndexX(varoffset);
            hypre_BoxIMinY(nbor_box) -= hypre_IndexY(varoffset);
            hypre_BoxIMinZ(nbor_box) -= hypre_IndexZ(varoffset);

            hypre_SStructNeighborPart(vneighbor) =
               hypre_SStructNeighborPart(neighbor);
            /* ZTODO: adjust ILower based on shift/subtract */
            hypre_CopyIndex(hypre_SStructNeighborILower(neighbor),
                            hypre_SStructNeighborILower(vneighbor));
            hypre_CopyIndex(hypre_SStructNeighborCoord(neighbor),
                            hypre_SStructNeighborCoord(vneighbor));
            hypre_CopyIndex(hypre_SStructNeighborDir(neighbor),
                            hypre_SStructNeighborDir(vneighbor));
         }
      }
   }

   hypre_SStructGridNVNeighbors(grid) = nvneighbors;
   hypre_SStructGridVNeighbors(grid)  = vneighbors;

   /*-------------------------------------------------
    * Assemble the map info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleMaps(grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridSetPeriodic( HYPRE_SStructGrid  grid,
                              int                part,
                              int               *periodic )
{
   int ierr = 0;

   hypre_SStructPGrid *pgrid          = hypre_SStructGridPGrid(grid, part);
   hypre_IndexRef      pgrid_periodic = hypre_SStructPGridPeriodic(pgrid);
   int                 d;

   for (d = 0; d < hypre_SStructGridNDim(grid); d++)
   {
      hypre_IndexD(pgrid_periodic, d) = periodic[d];
   }

   return ierr;
}

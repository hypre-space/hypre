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
 * HYPRE_SStructGridCreate
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
   hypre_SStructNeighbor ***neighbors;
   int                      i;

   grid = hypre_TAlloc(hypre_SStructGrid, 1);

   hypre_SStructGridComm(grid)   = comm;
   hypre_SStructGridNDim(grid)   = ndim;
   hypre_SStructGridNParts(grid) = nparts;
   pgrids = hypre_TAlloc(hypre_SStructPGrid *, nparts);
   neighbors = hypre_TAlloc(hypre_SStructNeighbor **, nparts);
   for (i = 0; i < nparts; i++)
   {
      hypre_SStructPGridCreate(comm, ndim, &pgrid);
      pgrids[i] = pgrid;
      neighbors[i] = NULL;
   }
   hypre_SStructGridPGrids(grid)  = pgrids;
   hypre_SStructGridNeighbors(grid) = neighbors;
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
 * HYPRE_SStructGridDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridDestroy( HYPRE_SStructGrid grid )
{
   int ierr = 0;

   int                        nparts;
   hypre_SStructPGrid       **pgrids;
   hypre_SStructNeighbor   ***neighbors;
   hypre_SStructNeighbor     *neighbor;
   hypre_BoxMap            ***maps;
   hypre_SStructBoxMapInfo ***info;
   int                        nvars;
   int                        part, var, i;

   if (grid)
   {
      hypre_SStructGridRefCount(grid) --;
      if (hypre_SStructGridRefCount(grid) == 0)
      {
         nparts  = hypre_SStructGridNParts(grid);
         pgrids  = hypre_SStructGridPGrids(grid);
         neighbors = hypre_SStructGridNeighbors(grid);
         maps = hypre_SStructGridMaps(grid);
         info = hypre_SStructGridInfo(grid);
         for (part = 0; part < nparts; part++)
         {
            if (neighbors[part] != NULL)
            {
               for (i = 0; i < nparts; i++)
               {
                  neighbor = neighbors[part][i];
                  hypre_BoxArrayDestroy(hypre_SStructNeighborBoxes(neighbor));
                  hypre_TFree(hypre_SStructNeighborILowers(neighbor));
               }
               hypre_TFree(neighbors[part]);
            }
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               hypre_BoxMapDestroy(maps[part][var]);
               hypre_TFree(info[part][var]);
            }
            hypre_SStructPGridDestroy(pgrids[part]);
            hypre_TFree(maps[part]);
            hypre_TFree(info[part]);
         }
         hypre_TFree(pgrids);
         hypre_TFree(neighbors);
         hypre_TFree(maps);
         hypre_TFree(info);
         hypre_TFree(grid);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridSetExtents
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
 * HYPRE_SStructGridSetVariables
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
 * HYPRE_SStructGridAddVariables
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
 * HYPRE_SStructGridSetNeighborBox
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

   int                      ndim      = hypre_SStructGridNDim(grid);
   hypre_SStructNeighbor ***neighbors = hypre_SStructGridNeighbors(grid);
   hypre_SStructNeighbor   *neighbor;
   hypre_BoxArray          *neighbor_boxes;
   hypre_Index             *neighbor_ilowers;
   int                      nboxes;

   hypre_Index              cilower;
   hypre_Index              ciupper;
   hypre_Box                box;
   int                      memchunk = 10;
   int                      d;

   if (neighbors[part] == NULL)
   {
      int nparts = hypre_SStructGridNParts(grid);
      int i;

      neighbors[part] = hypre_CTAlloc(hypre_SStructNeighbor *, nparts);
      for (i = 0; i < nparts; i++)
      {
         neighbors[part][i] = hypre_CTAlloc(hypre_SStructNeighbor, 1);
         hypre_SStructNeighborBoxes(neighbors[part][i]) =
            hypre_BoxArrayCreate(0);
      }
   }

   neighbor = neighbors[part][nbor_part];
   neighbor_boxes   = hypre_SStructNeighborBoxes(neighbor);
   neighbor_ilowers = hypre_SStructNeighborILowers(neighbor);
   nboxes = hypre_BoxArraySize(neighbor_boxes);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   hypre_BoxSetExtents(&box, cilower, ciupper);
   hypre_AppendBox(&box, neighbor_boxes);

   if ((nboxes % memchunk) == 0)
   {
      neighbor_ilowers = hypre_TReAlloc(neighbor_ilowers,
                                        hypre_Index, (nboxes + memchunk));
      hypre_SStructNeighborILowers(neighbor) = neighbor_ilowers;
   }
   hypre_CopyToCleanIndex(nbor_ilower, ndim, neighbor_ilowers[nboxes]);

   hypre_CopyToCleanIndex(index_map, ndim,
                          hypre_SStructNeighborCoord(neighbor));

   for (d = 0; d < ndim; d++)
   {
      if (hypre_IndexD(nbor_ilower, d) > hypre_IndexD(nbor_iupper, d))
      {
         hypre_IndexD(hypre_SStructNeighborDir(neighbor), d) = -1;
      }
      else
      {
         hypre_IndexD(hypre_SStructNeighborDir(neighbor), d) = 1;
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridAddUnstructuredPart *** placeholder ***
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
 * HYPRE_SStructGridAssemble
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGridAssemble( HYPRE_SStructGrid grid )
{
   int                   ierr = 0;

   int                      nparts    = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids    = hypre_SStructGridPGrids(grid);
   hypre_SStructNeighbor ***neighbors = hypre_SStructGridNeighbors(grid);
   /*hypre_SStructNeighbor   *neighbor;*/
   /*hypre_BoxArray          *neighbor_boxes;*/
   /*hypre_Index             *neighbor_ilowers;*/

   hypre_SStructPGrid      *pgrid;
   int                      part;

   /*-------------------------------------------------------------
    * use neighbor info to crop pgrids TODO
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      if (neighbors[part] != NULL)
      {
         /* set crop boxes for pgrid[part] */
      }
   }

   /*-------------------------------------------------------------
    * assemble the pgrids
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      if (pgrids[part] != NULL)
      {
         hypre_SStructPGridAssemble(pgrids[part]);
      }
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
    * Assemble the map info
    *-------------------------------------------------*/

   hypre_SStructGridAssembleMaps(grid);

   return ierr;
}


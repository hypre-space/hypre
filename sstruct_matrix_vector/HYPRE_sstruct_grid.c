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
   int                     *offsets;
   int                      i;

   grid = hypre_TAlloc(hypre_SStructGrid, 1);

   hypre_SStructGridComm(grid)   = comm;
   hypre_SStructGridNDim(grid)   = ndim;
   hypre_SStructGridNParts(grid) = nparts;
   pgrids  = hypre_TAlloc(hypre_SStructPGrid *, nparts);
   for (i = 0; i < nparts; i++)
   {
      hypre_SStructPGridCreate(comm, ndim, &pgrid);
      pgrids[i] = pgrid;
   }
   hypre_SStructGridPGrids(grid)  = pgrids;
   hypre_SStructGridNUCVars(grid) = 0;
   hypre_SStructGridUCVars(grid)  = NULL;

   offsets  = hypre_TAlloc(int, nparts);
   for (i = 0; i < nparts; i++)
   {
      offsets[i] = NULL;
   }
   hypre_SStructGridOffsets(grid)   = offsets;
   hypre_SStructGridUOffset(grid)   = 0;
   hypre_SStructGridStartRank(grid) = 0;

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

   int                  nparts;
   hypre_SStructPGrid **pgrids;
   int                  part;

   if (grid)
   {
      hypre_SStructGridRefCount(grid) --;
      if (hypre_SStructGridRefCount(grid) == 0)
      {
         nparts  = hypre_SStructGridNParts(grid);
         pgrids  = hypre_SStructGridPGrids(grid);
         hypre_TFree(hypre_SStructGridOffsets(grid));
         for (part = 0; part < nparts; part++)
         {
            hypre_SStructPGridDestroy(pgrids[part]);
         }
         hypre_TFree(pgrids);
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

   int                   nparts  = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid  **pgrids  = hypre_SStructGridPGrids(grid);
   int                  *offsets = hypre_SStructGridOffsets(grid);
   int                   uoffset;
   int                   start_rank;

   hypre_SStructPGrid   *pgrid;
   int                   part, offset;

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

   /*-------------------------------------------------
    * Set up the map info
    *-------------------------------------------------*/

   offset = 0;
   start_rank = 0;
   for (part = 0; part < nparts; part++)
   {
      offsets[part] = offset;
      offset += hypre_SStructPGridGlobalSize(pgrids[part]);
      start_rank += hypre_SStructPGridStartRank(pgrids[part]);
   }

   /* set up uoffset - TODO */
   uoffset = 0;

   hypre_SStructGridUOffset(grid)   = uoffset;
   hypre_SStructGridStartRank(grid) = start_rank + uoffset;

   /*-------------------------------------------------------------
    * set up the size info
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      hypre_SStructGridLocalSize(grid)  += hypre_SStructPGridLocalSize(pgrid);
      hypre_SStructGridGlobalSize(grid) += hypre_SStructPGridGlobalSize(pgrid);
   }

   return ierr;
}


/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructGrid interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructGrid
 *--------------------------------------------------------------------------*/

int
HYPRE_NewStructGrid( MPI_Comm          comm,
                     int               dim,
                     HYPRE_StructGrid *grid )
{
   *grid = ( (HYPRE_StructGrid) hypre_NewStructGrid( comm, dim ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructGrid
 *--------------------------------------------------------------------------*/

void 
HYPRE_FreeStructGrid( HYPRE_StructGrid grid )
{
   hypre_FreeStructGrid( (hypre_StructGrid *) grid );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
HYPRE_SetStructGridExtents( HYPRE_StructGrid  grid,
                            int              *ilower,
                            int              *iupper )
{
   hypre_Index  new_ilower;
   hypre_Index  new_iupper;

   int          d;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim((hypre_StructGrid *) grid); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }

   hypre_SetStructGridExtents( (hypre_StructGrid *) grid,
                               new_ilower, new_iupper );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void
HYPRE_AssembleStructGrid( HYPRE_StructGrid grid )
{
   hypre_AssembleStructGrid( (hypre_StructGrid *) grid, NULL, NULL, NULL );
}

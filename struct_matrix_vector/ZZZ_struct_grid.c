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
 * ZZZ_StructGrid interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructGrid
 *--------------------------------------------------------------------------*/

ZZZ_StructGrid
ZZZ_NewStructGrid( MPI_Comm *comm,
                   int       dim )
{
   return ( (ZZZ_StructGrid) zzz_NewStructGrid( comm, dim ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructGrid
 *--------------------------------------------------------------------------*/

void 
ZZZ_FreeStructGrid( ZZZ_StructGrid grid )
{
   zzz_FreeStructGrid( (zzz_StructGrid *) grid );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
ZZZ_SetStructGridExtents( ZZZ_StructGrid  grid,
                          int            *ilower,
                          int            *iupper )
{
   zzz_Index  new_ilower;
   zzz_Index  new_iupper;

   int        d;

   for (d = 0; d < zzz_StructGridDim((zzz_StructGrid *) grid); d++)
   {
      zzz_IndexD(new_ilower, d) = ilower[d];
      zzz_IndexD(new_iupper, d) = iupper[d];
   }
   for (d = zzz_StructGridDim((zzz_StructGrid *) grid); d < 3; d++)
   {
      zzz_IndexD(new_ilower, d) = 0;
      zzz_IndexD(new_iupper, d) = 0;
   }

   zzz_SetStructGridExtents( (zzz_StructGrid *) grid, new_ilower, new_iupper );
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
ZZZ_AssembleStructGrid( ZZZ_StructGrid grid )
{
   zzz_AssembleStructGrid( (zzz_StructGrid *) grid );
}



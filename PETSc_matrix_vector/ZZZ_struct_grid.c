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
ZZZ_NewStructGrid( int dim )
{
   return ( (ZZZ_StructGrid) zzz_NewStructGrid( dim ) );
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
		    int      *ilower,
		    int      *iupper )
{
   zzz_Index *new_ilower;
   zzz_Index *new_iupper;

   int        d;

   new_ilower = zzz_NewIndex();
   new_iupper = zzz_NewIndex();
   for (d = 0; d < zzz_StructGridDim((zzz_StructGrid *) grid); d++)
   {
      zzz_IndexD(new_ilower, d) = ilower[d];
      zzz_IndexD(new_iupper, d) = iupper[d];
   }

   zzz_SetStructGridExtents( (zzz_StructGrid *) grid, new_ilower, new_iupper );

   zzz_FreeIndex(new_ilower);
   zzz_FreeIndex(new_iupper);
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
ZZZ_AssembleStructGrid( ZZZ_StructGrid grid )
{
   zzz_AssembleStructGrid( (zzz_StructGrid *) grid );
}



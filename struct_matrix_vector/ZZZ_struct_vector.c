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
 * ZZZ_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructVector
 *--------------------------------------------------------------------------*/

ZZZ_StructVector
ZZZ_NewStructVector( MPI_Comm          *comm,
                     ZZZ_StructGrid     grid,
                     ZZZ_StructStencil  stencil )
{
   return ( (ZZZ_StructVector)
            zzz_NewStructVector( comm,
                                 (zzz_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_FreeStructVector( ZZZ_StructVector struct_vector )
{
   return( zzz_FreeStructVector( (zzz_StructVector *) struct_vector ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_InitializeStructVector
 *--------------------------------------------------------------------------*/

int
ZZZ_InitializeStructVector( ZZZ_StructVector vector )
{
   return ( zzz_InitializeStructVector( (zzz_StructVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructVectorValues( ZZZ_StructVector  vector,
                           int              *grid_index,
                           double            values     )
{
   zzz_StructVector *new_vector = (zzz_StructVector *) vector;
   zzz_Index        *new_grid_index;

   int               d;
   int               ierr;

   new_grid_index = zzz_NewIndex();
   for (d = 0;d < zzz_StructGridDim(zzz_StructVectorGrid(new_vector)); d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_SetStructVectorValues( new_vector, new_grid_index, values );

   zzz_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_GetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
ZZZ_GetStructVectorValues( ZZZ_StructVector  vector,
                           int              *grid_index,
                           double           *values     )
{
   zzz_StructVector *new_vector = (zzz_StructVector *) vector;
   zzz_Index        *new_grid_index;

   int               d;
   int               ierr;

   new_grid_index = zzz_NewIndex();
   for (d = 0;d < zzz_StructGridDim(zzz_StructVectorGrid(new_vector)); d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_GetStructVectorValues( new_vector, new_grid_index, values );

   zzz_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructVectorBoxValues( ZZZ_StructVector  vector,
                              int              *ilower,
                              int              *iupper,
                              int               num_stencil_indices,
                              int              *stencil_indices,
                              double           *values              )
{
   zzz_StructVector *new_vector = (zzz_StructVector *) vector;
   zzz_Index        *new_ilower;
   zzz_Index        *new_iupper;
   zzz_Box          *new_value_box;
                    
   int               d;
   int               ierr;

   new_ilower = zzz_NewIndex();
   new_iupper = zzz_NewIndex();
   for (d = 0;d < zzz_StructGridDim(zzz_StructVectorGrid(new_vector)); d++)
   {
      zzz_IndexD(new_ilower, d) = ilower[d];
      zzz_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = zzz_NewBox(new_ilower, new_iupper);

   ierr = zzz_SetStructVectorBoxValues( new_vector, new_value_box, values );

   zzz_FreeBox(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_AssembleStructVector( ZZZ_StructVector vector )
{
   return( zzz_AssembleStructVector( (zzz_StructVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_PrintStructVector
 *--------------------------------------------------------------------------*/

void
ZZZ_PrintStructVector( char            *filename,
                       ZZZ_StructVector vector,
                       int              all )
{
   zzz_PrintStructVector( filename,
			  (zzz_StructVector *) vector, 
			  all );
}

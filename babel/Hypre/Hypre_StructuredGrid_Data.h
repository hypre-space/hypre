/* *****************************************************
 *
 *	File:  Hypre_StructuredGrid_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructuredGrid_DataMembers_
#define Hypre_StructuredGrid_DataMembers_

#include "HYPRE_mv.h"
#include "struct_matrix_vector.h"
#include "struct_grid.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"

struct Hypre_StructuredGrid_private_type
{
   HYPRE_StructGrid *hsgrid;
   Hypre_MPI_Com comm;
   /* ... the MPI communicator information is also available from
      the HYPRE_StructGrid.  Putting it at this level makes a somewhat
      cleaner interface.  It's used in building a StructVector. */
}
;
#endif


/* *****************************************************
 *
 *	File:  Hypre_StructGrid_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructGrid_DataMembers_
#define Hypre_StructGrid_DataMembers_

#include "HYPRE_struct_mv.h"
#include "struct_mv.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"

struct Hypre_StructGrid_private_type
{
   HYPRE_StructGrid *hsgrid;
   Hypre_MPI_Com comm;
   /* ... the MPI communicator information is also available from
      the HYPRE_StructGrid.  Putting it at this level makes a somewhat
      cleaner interface.  It's used in building a StructVector. */

}
;
#endif


/* *****************************************************
 *
 *	File:  Hypre_StructVectorBuilder_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructVectorBuilder_DataMembers_
#define Hypre_StructVectorBuilder_DataMembers_

#include "Hypre_StructVector.h"

struct Hypre_StructVectorBuilder_private_type
{
   Hypre_StructVector newvec;
   /* ... the vector currently under construction */
   int vecgood;
   /* ... 1 if newvec is a valid vector, 0 if not, still under construction */
}
;
#endif


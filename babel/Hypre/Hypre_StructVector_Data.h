/* *****************************************************
 *
 *	File:  Hypre_StructVector_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructVector_DataMembers_
#define Hypre_StructVector_DataMembers_

#include "HYPRE_struct_mv.h"
#include "struct_mv.h"
#include "Hypre_StructGrid_IOR.h"

struct Hypre_StructVector_private_type
{
   HYPRE_StructVector *hsvec;
   /* We need to save constructor inputs to support a Clone function: */
   Hypre_StructGrid grid;
}
;
#endif


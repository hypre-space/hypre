/* *****************************************************
 *
 *	File:  Hypre_StructSMG_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructSMG_DataMembers_
#define Hypre_StructSMG_DataMembers_

#include "struct_ls.h"
#include "Hypre_StructMatrix_IOR.h"

struct Hypre_StructSMG_private_type
{
   HYPRE_StructSolver * hssolver;
   Hypre_StructMatrix hsmatrix;
}
;
#endif


/* *****************************************************
 *
 *	File:  Hypre_StructJacobi_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructJacobi_DataMembers_
#define Hypre_StructJacobi_DataMembers_

#include "struct_linear_solvers.h"
#include "Hypre_StructMatrix_IOR.h"

struct Hypre_StructJacobi_private_type
{
   HYPRE_StructSolver * hssolver;
   Hypre_StructMatrix hsmatrix;
}
;
#endif


/* *****************************************************
 *
 *	File:  Hypre_PCG_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_PCG_DataMembers_
#define Hypre_PCG_DataMembers_

#include "struct_linear_solvers.h"
#include "Hypre_StructMatrix.h"

struct Hypre_PCG_private_type
{
   HYPRE_StructSolver * hssolver;
   Hypre_StructMatrix hsmatrix;
}
;
#endif


/*#*****************************************************
#
#	File:  Hypre_StructJacobi_DataMembers.h
#
#********************************************************/

#ifndef Hypre_StructJacobi_DataMembers__
#define Hypre_StructJacobi_DataMembers__

/* JFP ... */
#include "struct_linear_solvers.h"
#include "Hypre_StructMatrix.h"

struct Hypre_StructJacobi_private_type /* gkk: added "_type" */
{
   HYPRE_StructSolver * hssolver;
   Hypre_StructMatrix hsmatrix;
};

#endif


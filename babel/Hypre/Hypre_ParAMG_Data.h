/* *****************************************************
 *
 *	File:  Hypre_ParAMG_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_ParAMG_DataMembers_
#define Hypre_ParAMG_DataMembers_

#include "Hypre_ParCSRMatrix_IOR.h"
#include "parcsr_linear_solvers.h"

struct Hypre_ParAMG_private_type
{
   HYPRE_Solver * Hsolver;
   Hypre_ParCSRMatrix Hmatrix;
}
;
#endif

